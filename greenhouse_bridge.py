"""
greenhouse_bridge.py
Мост между Java Monitoring Server и GreenLight симуляцией.

Функции:
    fetch_sensor_data(device_id, from_dt, to_dt)  — получить данные датчиков
    build_greenlight_csv(sensor_df)               — конвертировать в формат GreenLight
    run_simulation(csv_path)                      — запустить симуляцию
    compare(sim_df, sensor_df)                    — сравнить и найти аномалии
    run_pipeline(device_id, hours)                — полный пайплайн одной командой

Использование:
    python3 greenhouse_bridge.py --device-id 1 --hours 24

    from greenhouse_bridge import run_pipeline
    result = run_pipeline(device_id=1, hours=6)
"""

import sys
import os
import argparse
import tempfile
import warnings
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("greenhouse_bridge")

# ── GreenLight из установленного пакета ───────────────────────────────────────
_GL_SITE = "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages"
if _GL_SITE not in sys.path:
    sys.path.insert(0, _GL_SITE)

from greenlight import GreenLight  # noqa: E402

# ── Конфигурация ──────────────────────────────────────────────────────────────
API_BASE = os.environ.get("MONITORING_API", "http://localhost:8080")
GL_MAIN_JSON = os.path.join(
    _GL_SITE,
    "greenlight/models/katzin_2021/definition/main_katzin_2021.json",
)
ELEVATION_M = float(os.environ.get("GREENHOUSE_ELEVATION", "-5"))  # Нидерланды

# Порог Z-score для детекции аномалии
ANOMALY_ZSCORE = float(os.environ.get("ANOMALY_ZSCORE", "3.0"))

# ─────────────────────────────────────────────────────────────────────────────
# 1. ПОЛУЧЕНИЕ ДАННЫХ С МОНИТОРИНГОВОГО СЕРВЕРА
# ─────────────────────────────────────────────────────────────────────────────

def fetch_sensor_data(
    device_id: int,
    from_dt: datetime,
    to_dt: datetime,
    api_base: str = API_BASE,
) -> pd.DataFrame:
    """
    Запрашивает временной ряд показаний датчиков из мониторингового сервера.

    GET /api/sensors/{device_id}/range?from=...&to=...

    Параметры
    ---------
    device_id : int
        ID устройства в базе мониторингового сервера
    from_dt : datetime
        Начало периода
    to_dt : datetime
        Конец периода
    api_base : str
        Базовый URL мониторингового сервера (по умолчанию http://localhost:8080)

    Возвращает
    ----------
    pd.DataFrame с колонками: timestamp, airTemperature, airHumidity,
        co2Concentration, lightIntensity, parRadiation, outsideTemperature,
        outsideHumidity, windSpeed, soilTemperature, leafTemperature, ...
    """
    url = f"{api_base}/api/sensors/{device_id}/range"
    params = {
        "from": from_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "to":   to_dt.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    log.info(f"Запрос данных: {url}  {params}")
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Мониторинговый сервер недоступен по адресу {api_base}.\n"
            "Убедитесь, что Spring Boot запущен (mvn spring-boot:run в папке monitoringServer/)"
        )
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"HTTP ошибка: {e}")

    data = resp.json()
    if not data:
        raise ValueError(
            f"Нет данных для устройства {device_id} за период "
            f"{from_dt} — {to_dt}. Проверьте device_id и временной диапазон."
        )

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    log.info(f"Получено {len(df)} записей за период {df['timestamp'].iloc[0]} — {df['timestamp'].iloc[-1]}")
    return df


def fetch_latest(device_id: int, api_base: str = API_BASE) -> dict:
    """Возвращает последнее показание датчика (dict)."""
    url = f"{api_base}/api/sensors/{device_id}/latest"
    resp = requests.get(url, timeout=5)
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────────────────────────────────────
# 2. КОНВЕРТАЦИЯ В ФОРМАТ GREENLIGHT CSV
# ─────────────────────────────────────────────────────────────────────────────

def _vp_sat(T_C: np.ndarray) -> np.ndarray:
    """Давление насыщенного пара [Па] при температуре T [°C]."""
    return 610.78 * np.exp(17.27 * T_C / (T_C + 237.3))


def _rh_to_vp(rh_pct: np.ndarray, T_C: np.ndarray) -> np.ndarray:
    """Относительная влажность [%] + температура [°C] → давление пара [Па]."""
    return (rh_pct / 100.0) * _vp_sat(T_C)


def _daily_rad_sum(time_s: np.ndarray, iGlob: np.ndarray) -> np.ndarray:
    """
    Дневная сумма солнечной радиации [МДж м⁻²].
    Для каждой точки t — сумма iGlob за текущие сутки (0:00–t).
    """
    day_sum = np.zeros(len(time_s))
    days = time_s // 86400
    for d in np.unique(days):
        mask = days == d
        idxs = np.where(mask)[0]
        dt_s = np.diff(time_s[mask], prepend=time_s[mask][0])
        dt_s[0] = 0
        cumsum = np.cumsum(iGlob[mask] * dt_s) / 1e6  # Дж → МДж
        day_sum[idxs] = cumsum
    return day_sum


def build_greenlight_csv(
    sensor_df: pd.DataFrame,
    output_path: str | None = None,
) -> str:
    """
    Конвертирует DataFrame датчиков в CSV-файл формата GreenLight.

    Колонки на выходе (формат Bleiswijk):
        Time, tOut, vpOut, co2Out, wind, tSky, tSoOut,
        iGlob, dayRadSum, isDay, isDaySmooth, hElevation

    Параметры
    ---------
    sensor_df : pd.DataFrame
        Данные датчиков из fetch_sensor_data()
    output_path : str | None
        Путь для сохранения CSV. Если None — создаётся временный файл.

    Возвращает
    ----------
    str — путь к созданному CSV-файлу
    """
    df = sensor_df.copy()

    # ── Время в секундах от начала ────────────────────────────────────────────
    t0 = df["timestamp"].iloc[0]
    time_s = (df["timestamp"] - t0).dt.total_seconds().values

    # ── Уличная температура ────────────────────────────────────────────────────
    tOut = df.get("outsideTemperature", pd.Series(dtype=float)).values
    if np.isnan(tOut).any():
        # Фолбэк: немного ниже внутренней температуры
        tIn = df.get("airTemperature", pd.Series([20.0] * len(df))).fillna(20.0).values
        tOut = np.where(np.isnan(tOut), tIn - 5.0, tOut)

    # ── Давление пара снаружи [Па] ────────────────────────────────────────────
    outsideHumidity = df.get("outsideHumidity", pd.Series(dtype=float)).fillna(65.0).values
    vpOut = _rh_to_vp(outsideHumidity, tOut)

    # ── CO₂ снаружи [мг м⁻³]  (атм. уровень ~400 ppm = 784 мг/м³) ───────────
    co2Out_ppm = df.get("co2Concentration", pd.Series(dtype=float)).fillna(400.0).values
    co2Out = co2Out_ppm * 1.96  # ppm × 1.96 ≈ мг/м³ при стандартных условиях

    # ── Скорость ветра [м/с] ──────────────────────────────────────────────────
    wind = df.get("windSpeed", pd.Series(dtype=float)).fillna(2.0).values
    wind = np.where(np.isnan(wind), 2.0, wind)

    # ── Кажущаяся температура неба [°C] ─────────────────────────────────────
    # Оценка: tSky ≈ tOut - 20 (ясное небо); ночью тучи → ближе к tOut - 5
    iGlob_raw = df.get("lightIntensity", pd.Series(dtype=float)).fillna(0.0).values
    iGlob_raw = np.where(np.isnan(iGlob_raw), 0.0, iGlob_raw)
    # Перевод лк → Вт/м² (приблизительно: 1 Вт/м² ≈ 120 лк для дневного света)
    iGlob = iGlob_raw / 120.0

    # Если есть PAR (µmol m⁻² s⁻¹), уточняем: 1 Вт PAR ≈ 4.57 µmol m⁻² s⁻¹
    parRad = df.get("parRadiation", pd.Series(dtype=float)).values
    if not np.isnan(parRad).all():
        iGlob_from_par = np.where(np.isnan(parRad), iGlob, parRad / 4.57)
        iGlob = np.where(iGlob > 0, iGlob, iGlob_from_par)

    cloud_factor = np.where(iGlob > 10, 0.0, 1.0)  # облачность по iGlob
    tSky = tOut - 20 + 15 * cloud_factor  # [-20°C ясно, -5°C пасмурно]

    # ── Температура почвы на глубине 2 м [°C] ────────────────────────────────
    tSoOut = df.get("soilTemperature", pd.Series(dtype=float)).fillna(tOut.mean()).values
    tSoOut = np.where(np.isnan(tSoOut), tOut, tSoOut)

    # ── Дневная сумма радиации [МДж м⁻²] ─────────────────────────────────────
    dayRadSum = _daily_rad_sum(time_s, iGlob)

    # ── Признак дня ───────────────────────────────────────────────────────────
    isDay = (iGlob > 5).astype(float)
    from scipy.ndimage import gaussian_filter1d
    isDaySmooth = np.clip(gaussian_filter1d(isDay, sigma=4), 0, 1)

    # ── Высота над уровнем моря [м] ──────────────────────────────────────────
    hElevation = np.full(len(df), ELEVATION_M)

    # ── Сборка CSV ────────────────────────────────────────────────────────────
    out = pd.DataFrame({
        "Time":          time_s,
        "tOut":          tOut,
        "vpOut":         vpOut,
        "co2Out":        co2Out,
        "wind":          wind,
        "tSky":          tSky,
        "tSoOut":        tSoOut,
        "iGlob":         iGlob,
        "dayRadSum":     dayRadSum,
        "isDay":         isDay,
        "isDaySmooth":   isDaySmooth,
        "hElevation":    hElevation,
    })

    # Заголовки в формате GreenLight
    header_units = (
        "Time since start of data,Outdoor temperature,Outdoor vapor pressure,"
        "Outdoor CO2 concentration,Outdoor wind speed,Apparent sky temperature,"
        "Soil temperature at 2 m depth,Outdoor global solar radiation,"
        "Daily sum of outdoor global solar radiation,"
        "Switch determining if it is day or night. Used for control purposes,"
        "Smooth switch determining if it is day or night. Used for control purposes,"
        "Elevation at location"
    )
    header_si = "s,°C,Pa,mg m**-3,m s**-1,°C,°C,W m**-2,MJ m**-2,-,-,m above sea level"

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".csv", mode="w", delete=False, prefix="gl_bridge_"
        )
        output_path = tmp.name
        tmp.close()

    with open(output_path, "w") as f:
        f.write(",".join(out.columns) + "\n")
        f.write(header_units + "\n")
        f.write(header_si + "\n")
        out.to_csv(f, index=False, header=False)

    log.info(f"GreenLight CSV сохранён: {output_path}  ({len(out)} строк)")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# 3. ЗАПУСК СИМУЛЯЦИИ GREENLIGHT
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(
    csv_path: str,
    rtol: str = "1e-3",
    atol: str = "1e-1",
    output_step: int = 300,
) -> pd.DataFrame:
    """
    Запускает GreenLight симуляцию на данных из csv_path.

    Параметры
    ---------
    csv_path : str
        Путь к CSV-файлу, созданному build_greenlight_csv()
    rtol, atol : str
        Допуски ODE-решателя (BDF)
    output_step : int
        Шаг вывода в секундах (по умолчанию 300 с = 5 мин)

    Возвращает
    ----------
    pd.DataFrame — full_sol с колонками Time, tAir, tCan, vpAir, co2Air, ...
    """
    # Определяем длину симуляции по CSV
    df_in = pd.read_csv(csv_path, skiprows=2)
    t_end = int(df_in.iloc[-1, 0])  # первая колонка — Time [с]

    log.info(f"Запуск симуляции: T_end={t_end} с ({t_end/3600:.1f} ч)")
    mdl = GreenLight(input_prompt=[GL_MAIN_JSON, csv_path])
    mdl.options["t_end"]       = str(t_end)
    mdl.options["output_step"] = str(output_step)
    mdl.options["rtol"]        = rtol
    mdl.options["atol"]        = atol

    mdl.load()
    mdl.solve()

    df = mdl.full_sol.copy()
    log.info(f"Симуляция завершена. Строк: {len(df)}, переменных: {len(df.columns)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. СРАВНЕНИЕ: МОДЕЛЬ VS ДАТЧИКИ + ДЕТЕКЦИЯ АНОМАЛИЙ
# ─────────────────────────────────────────────────────────────────────────────

def compare(
    sim_df: pd.DataFrame,
    sensor_df: pd.DataFrame,
    t0: datetime,
    zscore_threshold: float = ANOMALY_ZSCORE,
) -> pd.DataFrame:
    """
    Совмещает результаты симуляции с реальными показаниями датчиков.
    Вычисляет ошибки и Z-score для каждой сравниваемой переменной.

    Параметры
    ---------
    sim_df : pd.DataFrame
        Результат run_simulation() — содержит Time [с] и переменные модели
    sensor_df : pd.DataFrame
        Результат fetch_sensor_data() — содержит timestamp и поля датчиков
    t0 : datetime
        Начальный момент симуляции (для перевода Time [с] → datetime)
    zscore_threshold : float
        Порог |Z-score| для пометки аномалии (по умолчанию 3.0)

    Возвращает
    ----------
    pd.DataFrame с колонками:
        timestamp,
        tAir_sim, tAir_real, delta_tAir, zscore_tAir,
        RH_sim,   RH_real,   delta_RH,   zscore_RH,
        co2_sim,  co2_real,  delta_co2,  zscore_co2,
        anomaly (bool)
    """
    # ── Переводим время симуляции в datetime ──────────────────────────────────
    sim = sim_df.copy()
    sim["timestamp"] = pd.to_datetime(
        [t0 + timedelta(seconds=float(s)) for s in sim["Time"]]
    )
    sim = sim.set_index("timestamp")

    # ── Ресемплируем датчики на сетку симуляции ───────────────────────────────
    sens = sensor_df.copy().set_index("timestamp")
    # Интерполяция на временную сетку симуляции
    sens_resampled = sens.reindex(
        sens.index.union(sim.index)
    ).interpolate("time").reindex(sim.index)

    # ── Вычисляем сравниваемые переменные ────────────────────────────────────
    result = pd.DataFrame(index=sim.index)
    result.index.name = "timestamp"

    # 1. Температура воздуха
    if "tAir" in sim.columns and "airTemperature" in sens_resampled.columns:
        result["tAir_sim"]  = sim["tAir"]
        result["tAir_real"] = sens_resampled["airTemperature"]
        result["delta_tAir"] = result["tAir_sim"] - result["tAir_real"]
        result["zscore_tAir"] = _zscore(result["delta_tAir"])

    # 2. Относительная влажность
    if "vpAir" in sim.columns and "airHumidity" in sens_resampled.columns:
        vp_sat_arr = _vp_sat(sim["tAir"].values if "tAir" in sim.columns else np.full(len(sim), 20.0))
        result["RH_sim"]  = np.clip(sim["vpAir"].values / vp_sat_arr * 100, 0, 100)
        result["RH_real"] = sens_resampled["airHumidity"]
        result["delta_RH"] = result["RH_sim"] - result["RH_real"]
        result["zscore_RH"] = _zscore(result["delta_RH"])

    # 3. CO₂ (переводим модель мг/м³ → ppm для сравнения)
    if "co2Air" in sim.columns and "co2Concentration" in sens_resampled.columns:
        result["co2_sim"]  = sim["co2Air"] / 1.96  # мг/м³ → ppm
        result["co2_real"] = sens_resampled["co2Concentration"]
        result["delta_co2"] = result["co2_sim"] - result["co2_real"]
        result["zscore_co2"] = _zscore(result["delta_co2"])

    # ── Флаг аномалии ─────────────────────────────────────────────────────────
    z_cols = [c for c in result.columns if c.startswith("zscore_")]
    if z_cols:
        result["anomaly"] = result[z_cols].abs().max(axis=1) > zscore_threshold
    else:
        result["anomaly"] = False

    n_anomaly = result["anomaly"].sum()
    pct = n_anomaly / len(result) * 100
    log.info(f"Аномалий: {n_anomaly} из {len(result)} точек ({pct:.1f}%)")

    return result.reset_index()


def _zscore(series: pd.Series) -> pd.Series:
    """Z-score нормализация (скользящее среднее ±30 мин)."""
    mu  = series.rolling(window=12, min_periods=1, center=True).mean()
    std = series.rolling(window=12, min_periods=1, center=True).std().fillna(1.0)
    std = std.replace(0, 1.0)
    return (series - mu) / std


# ─────────────────────────────────────────────────────────────────────────────
# 5. ПОЛНЫЙ ПАЙПЛАЙН
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    device_id: int,
    hours: float = 24,
    to_dt: datetime | None = None,
    api_base: str = API_BASE,
) -> dict:
    """
    Полный пайплайн: датчики → GreenLight → сравнение.

    Параметры
    ---------
    device_id : int
        ID устройства в мониторинговом сервере
    hours : float
        Сколько часов назад берём данные (по умолчанию 24)
    to_dt : datetime | None
        Конец периода (по умолчанию — сейчас)
    api_base : str
        URL мониторингового сервера

    Возвращает
    ----------
    dict с ключами:
        'sensor_df'   — сырые данные датчиков
        'sim_df'      — результаты симуляции GreenLight
        'compare_df'  — сравнение с аномалиями
        'csv_path'    — путь к временному CSV-файлу
        'from_dt'     — начало периода
        'to_dt'       — конец периода
    """
    if to_dt is None:
        to_dt = datetime.now()
    from_dt = to_dt - timedelta(hours=hours)

    log.info(f"═══ Пайплайн: device={device_id}, {from_dt} — {to_dt} ═══")

    # Шаг 1: Получить данные датчиков
    sensor_df = fetch_sensor_data(device_id, from_dt, to_dt, api_base)

    # Шаг 2: Конвертировать в CSV
    csv_path = build_greenlight_csv(sensor_df)

    # Шаг 3: Запустить симуляцию
    sim_df = run_simulation(csv_path)

    # Шаг 4: Сравнить
    compare_df = compare(sim_df, sensor_df, t0=sensor_df["timestamp"].iloc[0])

    log.info("═══ Готово ═══")
    return {
        "sensor_df":  sensor_df,
        "sim_df":     sim_df,
        "compare_df": compare_df,
        "csv_path":   csv_path,
        "from_dt":    from_dt,
        "to_dt":      to_dt,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(
        description="GreenLight ↔ Monitoring Server bridge"
    )
    parser.add_argument("--device-id", type=int, default=1,
                        help="ID устройства (default: 1)")
    parser.add_argument("--hours", type=float, default=24,
                        help="Сколько часов анализировать (default: 24)")
    parser.add_argument("--api", default=API_BASE,
                        help=f"URL мониторингового сервера (default: {API_BASE})")
    parser.add_argument("--output", default=None,
                        help="Путь для сохранения compare_df в CSV")
    args = parser.parse_args()

    try:
        result = run_pipeline(
            device_id=args.device_id,
            hours=args.hours,
            api_base=args.api,
        )
    except (RuntimeError, ValueError) as e:
        log.error(str(e))
        sys.exit(1)

    cdf = result["compare_df"]

    print("\n📊 Результат сравнения (первые 5 строк):")
    print(cdf.head().to_string(index=False))

    print(f"\n⚠️  Аномалий обнаружено: {cdf['anomaly'].sum()} из {len(cdf)} точек")

    if args.output:
        cdf.to_csv(args.output, index=False)
        log.info(f"Сохранено: {args.output}")
    else:
        tmp_out = tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, prefix="gl_compare_"
        ).name
        cdf.to_csv(tmp_out, index=False)
        log.info(f"Результат сравнения: {tmp_out}")


if __name__ == "__main__":
    _cli()
