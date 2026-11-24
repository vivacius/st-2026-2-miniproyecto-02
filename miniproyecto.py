"""Miniproyecto 2: aprendizaje automático y profundo contra el mejor modelo clásico.

Se corre de principio a fin con

    python miniproyecto.py

y deja las figuras en figuras/ y las tablas en resultados/. Las partes
marcadas con TODO las completa el grupo; mientras falten, el guion las
reporta como pendientes y sigue. El informe, docs/informe.md, cita las cifras
y las figuras que produce este guion.
"""
import math
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402
from statsmodels.tsa.statespace.sarimax import SARIMAX  # noqa: E402
from torch import nn  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from comun import (RAIZ, backtest, cargar_serie, estacional_ingenuo,  # noqa: E402
                   fechas_siguientes, rmse)

# Después de importar statsmodels, que al cargarse fija sus propios avisos.
warnings.filterwarnings("ignore")
torch.set_num_threads(1)
FIGURAS = RAIZ / "figuras"
RESULTADOS = RAIZ / "resultados"
FIGURAS.mkdir(exist_ok=True)
RESULTADOS.mkdir(exist_ok=True)

SEMILLA = 2026
N_PRUEBA = 24          # los meses del backtest, los mismos de comun.backtest

# El último modelo que ajustó cada función modelo(entrenamiento, h). Sirve
# para contar sus parámetros después del backtest.
ULTIMO = {}

# ============================================================
# 1. La serie y el modelo a superar
# ============================================================
ARCHIVO = "datos/ventas_mensuales.csv"
FECHA = "observation_date"
VALOR = "MRTSSM452USN"
PERIODO = 12


def preparar(serie):
    """Misma preparación del miniproyecto 1: sin faltantes ni extremos alterados."""
    serie = serie.copy()
    if serie.isna().any():
        raise ValueError("Se encontraron valores faltantes no previstos")
    if not np.isfinite(serie.to_numpy()).all():
        raise ValueError("Existen valores no finitos")
    return serie


def exogenas(serie):
    """Variable de calendario conocida anticipadamente: días hábiles (lun-vie) del mes."""
    fechas = pd.date_range(
        start=serie.index.min(),
        end=serie.index.max() + pd.DateOffset(months=12),
        freq="MS",
    )
    dias = [
        np.busday_count(fecha.date(), (fecha + pd.offsets.MonthBegin(1)).date())
        for fecha in fechas
    ]
    x = pd.DataFrame({"dias_lunes_viernes": dias}, index=fechas)
    x.index.name = "fecha"
    return x


def ajustar(modelo, etapa, origen):
    """Ajusta el SARIMAX con maxiter=100 y verifica convergencia sin descartar orígenes."""
    ajuste = modelo.fit(disp=False, maxiter=100)
    convergio = bool(ajuste.mle_retvals.get("converged", False))
    if not convergio:
        raise RuntimeError(
            f"{etapa} no convergió en {origen:%Y-%m}; se detiene la ejecución "
            "sin descartar el origen ni sustituir el modelo."
        )
    return ajuste


def modelo_sarimax(orden, orden_estacional, x):
    """SARIMAX con exógena de calendario, sin fuga de información."""
    def pronosticar(entrenamiento, h):
        x_entrenamiento = x.loc[entrenamiento.index]
        fechas_futuras = fechas_siguientes(entrenamiento, h)
        x_futuro = x.loc[fechas_futuras]
        modelo = SARIMAX(
            entrenamiento,
            exog=x_entrenamiento,
            order=orden,
            seasonal_order=orden_estacional,
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        ajuste = ajustar(modelo, "backtest SARIMAX", entrenamiento.index[-1])
        ULTIMO["modelo"] = ajuste
        return ajuste.forecast(steps=h, exog=x_futuro).to_numpy()

    return pronosticar


def modelo_clasico(serie):
    """Mejor modelo del miniproyecto 1: SARIMAX(1, 0, 2)(1, 1, 1, 12) + días lun-vie."""
    x = exogenas(serie)
    return modelo_sarimax((1, 0, 2), (1, 1, 1, PERIODO), x)


NOMBRE_CLASICO = "SARIMAX(1,0,2)(1,1,1,12) + exógenas"


def graficar_serie(serie):
    figura, eje = plt.subplots(figsize=(9.5, 3.6))
    eje.plot(serie.index, serie.to_numpy(), color="#1f4e79", linewidth=1.3, label="Ventas mensuales")
    eje.axvline(serie.index[-N_PRUEBA], color="#b22222", linestyle="--", linewidth=1.2,
                label="Inicio backtest (24 meses)")
    eje.set_title(f"{VALOR}: ventas mensuales en EE. UU. (1992–2026)")
    eje.set_xlabel("Fecha")
    eje.set_ylabel("Millones de USD")
    eje.grid(True, alpha=0.25)
    eje.spines["top"].set_visible(False)
    eje.spines["right"].set_visible(False)
    eje.legend(frameon=False, fontsize=9)
    figura.tight_layout()
    figura.savefig(FIGURAS / "serie.png", dpi=150)
    plt.close(figura)


# ============================================================
# 2. La tabla supervisada
# ============================================================
# Lo que aprende cada modelo: "nivel" es la serie tal cual; "dif12" es la
# diferencia con el mismo mes del año anterior.
OBJETIVO = "dif12"
REZAGOS = 12


def objetivo(y):
    return y.diff(PERIODO) if OBJETIVO == "dif12" else y


def a_nivel(y, z):
    """Del pronóstico del objetivo al nivel: y(t) = y(t-12) + z(t) si z es la diferencia."""
    return y.iloc[-PERIODO] + z if OBJETIVO == "dif12" else z


def tabla_supervisada(z):
    """Una fila por mes con rezagos 1 a 12, mes del año y media móvil de 3 meses previos."""
    X = pd.DataFrame(index=z.index)
    for k in range(1, REZAGOS + 1):
        X[f"rezago_{k}"] = z.shift(k)
    X["mes"] = z.index.month
    X["media_3"] = z.shift(1).rolling(3).mean()
    return X


def comprobar_sin_fuga(serie):
    """Prueba exhaustiva de que ninguna fila ve el mes que predice."""
    z = objetivo(serie)
    X = tabla_supervisada(z)
    X_comp, z_comp = filas_completas(serie)

    # Verificar que alterar cada mes t no modifica su propia fila X.iloc[t]
    for i in range(len(serie)):
        serie_alt = serie.copy()
        serie_alt.iloc[i] = serie_alt.iloc[i] + 12345.67
        X_alt = tabla_supervisada(objetivo(serie_alt))
        assert np.array_equal(
            X.iloc[i].to_numpy(dtype=float),
            X_alt.iloc[i].to_numpy(dtype=float),
            equal_nan=True,
        ), f"Fuga detectada en la fila {serie.index[i]:%Y-%m} al alterar la serie"

        z_alt = z.copy()
        z_alt.iloc[i] = (0.0 if pd.isna(z_alt.iloc[i]) else z_alt.iloc[i]) + 12345.67
        X_z_alt = tabla_supervisada(z_alt)
        assert np.array_equal(
            X.iloc[i].to_numpy(dtype=float),
            X_z_alt.iloc[i].to_numpy(dtype=float),
            equal_nan=True,
        ), f"Fuga detectada en la fila {serie.index[i]:%Y-%m} al alterar el objetivo"

    X_ent, z_ent = filas_completas(serie.iloc[:-N_PRUEBA])
    print(f"  Filas iniciales: {len(X)}, completas: {len(X_comp)}, "
          f"perdidas: {len(X) - len(X_comp)} (12 por diff(12) + 12 por rezago_12)")
    print(f"  Columnas ({len(X.columns)}): {list(X.columns)}")
    print(f"  Dimensiones entrenamiento previo al backtest: X={X_ent.shape}, z={z_ent.shape}")
    print(f"  Estadísticas objetivo z (previo backtest): media={z_ent.mean():.2f}, "
          f"desv={z_ent.std():.2f}, min={z_ent.min():.2f}, max={z_ent.max():.2f}")
    muestra_mes = X_comp.index[-1]
    print(f"  Ejemplo fila {muestra_mes:%Y-%m}: "
          f"rezago_1={X_comp.loc[muestra_mes, 'rezago_1']:.2f}, "
          f"rezago_12={X_comp.loc[muestra_mes, 'rezago_12']:.2f}, "
          f"mes={int(X_comp.loc[muestra_mes, 'mes'])}, "
          f"media_3={X_comp.loc[muestra_mes, 'media_3']:.2f} | "
          f"z_t={z_comp.loc[muestra_mes]:.2f}")
    print(f"  Prueba sin fuga: OK ({len(serie)}/{len(serie)} meses verificados sin cambios)")


def filas_completas(y):
    z = objetivo(y)
    X = tabla_supervisada(z)
    completas = X.notna().all(axis=1) & z.notna()
    return X[completas], z[completas]


def recursivo(ajustado, y, h):
    """Pronóstico recursivo a h meses: cada predicción entra como rezago del mes siguiente."""
    y_ext = y.copy()
    for _ in range(h):
        siguiente = y_ext.index[-1] + pd.offsets.MonthBegin(1)
        y_ext.loc[siguiente] = np.nan
        X_fut = tabla_supervisada(objetivo(y_ext)).iloc[[-1]]
        z_hat = float(ajustado.predict(X_fut)[0])
        y_ext.iloc[-1] = a_nivel(y_ext.iloc[:-1], z_hat)
    return y_ext.iloc[-h:].to_numpy()


def con_tabla(fabrica):
    """Modelo para el backtest: arma la tabla con el entrenamiento del origen,
    ajusta un modelo nuevo y pronostica h meses en modo recursivo. El
    escalador vive dentro del modelo, así que se ajusta en cada origen."""
    def modelo(entrenamiento, h):
        X, z = filas_completas(entrenamiento)
        ajustado = fabrica()
        ajustado.fit(X, z)
        ULTIMO["modelo"] = ajustado
        return recursivo(ajustado, entrenamiento, h)
    return modelo


# ============================================================
# 3. Aprendizaje automático
# ============================================================
# Las cuatro fábricas devuelven un modelo nuevo sin entrenar; con_tabla
# ajusta uno por origen. KNN y SVR llevan el escalador dentro del pipeline.

def knn(n_neighbors=5):
    return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=n_neighbors))


def svr(C=1.0, epsilon=0.1):
    return make_pipeline(StandardScaler(), SVR(kernel="rbf", C=C, epsilon=epsilon))


def bosque(max_depth=None, min_samples_leaf=1):
    return RandomForestRegressor(n_estimators=100, max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf, random_state=SEMILLA)


def xgboost(max_depth=3, learning_rate=0.1):
    return XGBRegressor(n_estimators=200, max_depth=max_depth, learning_rate=learning_rate,
                        random_state=SEMILLA)


REJILLAS = {
    "KNN": (knn, {"n_neighbors": [3, 5, 9]}),
    "SVR": (svr, {"C": [100, 1000, 10000], "epsilon": [10, 100]}),
    "Random Forest": (bosque, {"max_depth": [4, 8, None], "min_samples_leaf": [1, 4]}),
    "XGBoost": (xgboost, {"max_depth": [2, 4, 6], "learning_rate": [0.03, 0.1]}),
}


def sin_prefijo(parametros):
    return {nombre.split("__")[-1]: valor for nombre, valor in parametros.items()}


def ajustar_hiperparametros(serie):
    """Para cada modelo de REJILLAS, selecciona hiperparámetros con TimeSeriesSplit."""
    X, z = filas_completas(serie.iloc[:-N_PRUEBA])
    tscv = TimeSeriesSplit(n_splits=5)

    print("  Pliegues TimeSeriesSplit (sin tocar los 24 meses del backtest):")
    for k, (idx_ent, idx_val) in enumerate(tscv.split(X), start=1):
        print(f"    Pliegue {k}: ent {X.index[idx_ent[0]]:%Y-%m}..{X.index[idx_ent[-1]]:%Y-%m} "
              f"({len(idx_ent)}) -> val {X.index[idx_val[0]]:%Y-%m}..{X.index[idx_val[-1]]:%Y-%m} "
              f"({len(idx_val)})")

    elegidos = {}
    registros = []
    for nombre, (fabrica, rejilla) in REJILLAS.items():
        t0 = time.perf_counter()
        base = fabrica()
        if hasattr(base, "steps"):
            paso = base.steps[-1][0]
            rejilla_cv = {f"{paso}__{k}": v for k, v in rejilla.items()}
        else:
            rejilla_cv = rejilla

        busqueda = GridSearchCV(
            base,
            rejilla_cv,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        busqueda.fit(X, z)
        duracion = time.perf_counter() - t0
        mejores = sin_prefijo(busqueda.best_params_)
        rmse_val = -busqueda.best_score_
        elegidos[nombre] = mejores
        registros.append({
            "modelo": nombre,
            "rejilla": str(rejilla),
            "mejores_parametros": str(mejores),
            "rmse_validacion": round(rmse_val, 2),
            "segundos_busqueda": round(duracion, 2),
        })
        print(f"  {nombre}: mejores={mejores}, RMSE validación={rmse_val:.2f} ({duracion:.1f} s)")

    pd.DataFrame(registros).to_csv(RESULTADOS / "ml_validacion.csv", index=False)
    return elegidos


def graficar_importancia(serie, elegidos):
    """Importancia de las variables de Random Forest y XGBoost sobre el entrenamiento previo."""
    X, z = filas_completas(serie.iloc[:-N_PRUEBA])
    rf = bosque(**elegidos["Random Forest"]).fit(X, z)
    xgb = xgboost(**elegidos["XGBoost"]).fit(X, z)

    imp_rf = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
    imp_xgb = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=True)

    print("  Top 5 variables Random Forest:",
          {k: round(float(v), 4) for k, v in imp_rf.iloc[::-1].head(5).items()})
    print("  Top 5 variables XGBoost:",
          {k: round(float(v), 4) for k, v in imp_xgb.iloc[::-1].head(5).items()})

    figura, ejes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for eje, imp, titulo, color in [
        (ejes[0], imp_rf, "Random Forest", "#2e6f40"),
        (ejes[1], imp_xgb, "XGBoost", "#d95f02"),
    ]:
        eje.barh(imp.index, imp.to_numpy(), color=color, alpha=0.88)
        eje.set_title(f"Importancia de variables: {titulo}")
        eje.set_xlabel("Importancia relativa")
        eje.grid(True, axis="x", alpha=0.25)
        eje.spines["top"].set_visible(False)
        eje.spines["right"].set_visible(False)

    figura.tight_layout()
    figura.savefig(FIGURAS / "importancia.png", dpi=150)
    plt.close(figura)


# ============================================================
# 4. Aprendizaje profundo en PyTorch
# ============================================================
# Redes pequeñas y pocas épocas: el guion completo tiene que correr en CPU
# en pocos minutos.
EPOCAS_MAXIMAS = 300
TASA = 0.01
# En Colab con GPU la red y sus tensores van a cuda; sin GPU, a la CPU. Red se
# encarga de los dos: las clases de cada red no tienen que mover nada.
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"


class Red:
    """Una red de PyTorch con fit y predict, para entrar a la tabla como un
    modelo de scikit-learn. Escala X y z con lo que ve en fit, que es el
    entrenamiento del origen, y devuelve el pronóstico en la escala original.
    Con secuencia=True la red recibe los rezagos en orden, con forma
    (lote, 12, 1), del más viejo al más reciente."""

    def __init__(self, construir, secuencia=False, epocas=EPOCAS_MAXIMAS, semilla=SEMILLA):
        self.construir = construir
        self.secuencia = secuencia
        self.epocas = epocas
        self.semilla = semilla

    def entradas(self, X):
        if self.secuencia:
            columnas = [f"rezago_{k}" for k in range(REZAGOS, 0, -1)]
            x = (X[columnas].to_numpy() - self.media) / self.desv
            return torch.tensor(x[:, :, None], dtype=torch.float32, device=DISPOSITIVO)
        return torch.tensor(self.escala_x.transform(X), dtype=torch.float32, device=DISPOSITIVO)

    def fit(self, X, z, X_val=None, z_val=None):
        torch.manual_seed(self.semilla)
        self.media, self.desv = z.mean(), z.std()
        self.escala_x = StandardScaler().fit(X)
        x = self.entradas(X)
        objetivo_t = torch.tensor(((z - self.media) / self.desv).to_numpy()[:, None],
                                  dtype=torch.float32, device=DISPOSITIVO)
        self.red = self.construir(x.shape[-1]).to(DISPOSITIVO)
        optimizador = torch.optim.Adam(self.red.parameters(), lr=TASA)
        perdida = nn.MSELoss()
        self.perdidas = []
        for _ in range(self.epocas):
            self.red.train()
            optimizador.zero_grad()
            error = perdida(self.red(x), objetivo_t)
            error.backward()
            optimizador.step()
            fila = [error.item()]
            if X_val is not None:
                fila.append(float(np.mean(((self.predict(X_val) - z_val) / self.desv) ** 2)))
            self.perdidas.append(fila)
        return self

    def predict(self, X):
        self.red.eval()
        with torch.no_grad():
            salida = self.red(self.entradas(X)).cpu().numpy()[:, 0]
        return salida * self.desv + self.media


class MLP(nn.Module):
    def __init__(self, entradas, ocultas=16):
        super().__init__()
        self.capas = nn.Sequential(nn.Linear(entradas, ocultas), nn.ReLU(),
                                   nn.Linear(ocultas, 1))

    def forward(self, x):
        return self.capas(x)


class LSTM(nn.Module):
    """Una capa LSTM sobre la secuencia (lote, 12, 1) y salida lineal del último estado."""

    def __init__(self, ocultas=32):
        super().__init__()
        self.celda = nn.LSTM(1, ocultas, num_layers=1, batch_first=True)
        self.salida = nn.Linear(ocultas, 1)

    def forward(self, x):
        estados, _ = self.celda(x)
        return self.salida(estados[:, -1, :])


def codificacion_posicion(pasos, dimension):
    """Codificación posicional sinusoidal para la ventana temporal."""
    posicion = torch.arange(pasos, dtype=torch.float32).unsqueeze(1)
    frecuencia = torch.exp(
        torch.arange(0, dimension, 2, dtype=torch.float32) * (-math.log(10000.0) / dimension)
    )
    tabla = torch.zeros(pasos, dimension)
    tabla[:, 0::2] = torch.sin(posicion * frecuencia)
    tabla[:, 1::2] = torch.cos(posicion * frecuencia)
    return tabla


class Transformer(nn.Module):
    """Proyección de entrada, posición sinusoidal, 1 capa encoder y salida lineal."""

    def __init__(self, variables=1, dimension=16, cabezas=2, pasos=12, ancho=32):
        super().__init__()
        self.entrada = nn.Linear(variables, dimension)
        self.register_buffer("posicion", codificacion_posicion(pasos, dimension))
        self.capa = nn.TransformerEncoderLayer(
            dimension, cabezas, dim_feedforward=ancho, dropout=0.0, batch_first=True
        )
        self.salida = nn.Linear(dimension, 1)

    def forward(self, x):
        h = self.entrada(x) + self.posicion[:x.shape[1]]
        return self.salida(self.capa(h)[:, -1, :])


# Las candidatas: nombre, cómo se construye y si recibe la secuencia.
CANDIDATAS = {
    "MLP (16 ocultas)": (lambda n: MLP(n, 16), False),
    "MLP (32 ocultas)": (lambda n: MLP(n, 32), False),
    "LSTM (32 ocultas)": (lambda _: LSTM(32), True),
    "Transformer (d=16, 2 cabezas)": (
        lambda n: Transformer(n, dimension=16, cabezas=2, pasos=REZAGOS, ancho=32),
        True,
    ),
}


def validar_redes(serie):
    """Elige el tamaño del MLP y las épocas de cada red con validación temporal previa."""
    X, z = filas_completas(serie.iloc[:-N_PRUEBA])
    n_val = 24
    X_ent, z_ent = X.iloc[:-n_val], z.iloc[:-n_val]
    X_val, z_val = X.iloc[-n_val:], z.iloc[-n_val:]
    print(f"  Validación redes: ent {X_ent.index[0]:%Y-%m}..{X_ent.index[-1]:%Y-%m} ({len(X_ent)}), "
          f"val {X_val.index[0]:%Y-%m}..{X_val.index[-1]:%Y-%m} ({len(X_val)})")

    historias = {}
    resumen = []
    for nombre, (construir, secuencia) in CANDIDATAS.items():
        red_tmp = Red(construir, secuencia=secuencia, epocas=EPOCAS_MAXIMAS, semilla=SEMILLA)
        red_tmp.fit(X_ent, z_ent, X_val, z_val)
        perdidas = np.asarray(red_tmp.perdidas)
        epoca_opt = int(np.argmin(perdidas[:, 1])) + 1
        min_val = float(perdidas[epoca_opt - 1, 1])
        params = contar_parametros(red_tmp)
        historias[nombre] = (perdidas, epoca_opt, min_val)
        resumen.append({
            "red": nombre,
            "parametros": params,
            "epoca_optima": epoca_opt,
            "perdida_val_min": round(min_val, 4),
            "perdida_ent_final": round(float(perdidas[-1, 0]), 4),
        })
        print(f"  {nombre}: params={params}, época óptima={epoca_opt}, "
              f"MSE val mín={min_val:.4f}")

    # Graficar curvas de entrenamiento y validación (2x2)
    figura, ejes = plt.subplots(2, 2, figsize=(10.5, 6.8), sharex=True)
    for eje, (nombre, (perdidas, epoca_opt, min_val)) in zip(ejes.ravel(), historias.items()):
        epocas_eje = np.arange(1, len(perdidas) + 1)
        eje.plot(epocas_eje, perdidas[:, 0], label="Entrenamiento", color="#1f4e79", linewidth=1.3)
        eje.plot(epocas_eje, perdidas[:, 1], label="Validación", color="#d95f02", linewidth=1.3)
        eje.axvline(epoca_opt, color="#2e6f40", linestyle="--", linewidth=1.1,
                    label=f"Óptima: {epoca_opt} ({min_val:.3f})")
        eje.set_title(nombre, fontsize=10)
        eje.set_ylabel("MSE escalado")
        eje.grid(True, alpha=0.25)
        eje.spines["top"].set_visible(False)
        eje.spines["right"].set_visible(False)
        eje.legend(frameon=False, fontsize=8)
    ejes[1, 0].set_xlabel("Época")
    ejes[1, 1].set_xlabel("Época")
    figura.suptitle("Curvas de pérdida de entrenamiento y validación (sin tocar el backtest)", fontsize=11)
    figura.tight_layout()
    figura.savefig(FIGURAS / "perdida.png", dpi=150)
    plt.close(figura)

    # Elegir el mejor MLP por pérdida mínima de validación
    mlps = [r for r in resumen if r["red"].startswith("MLP")]
    mejor_mlp = min(mlps, key=lambda r: r["perdida_val_min"])["red"]
    print(f"  MLP elegido por validación: {mejor_mlp}")

    elegidas = {}
    for r in resumen:
        nombre = r["red"]
        if nombre.startswith("MLP") and nombre != mejor_mlp:
            r["evaluada_backtest"] = False
            continue
        r["evaluada_backtest"] = True
        elegidas[nombre] = r["epoca_optima"]

    pd.DataFrame(resumen).to_csv(RESULTADOS / "redes_validacion.csv", index=False)
    return elegidas


def red(nombre, epocas, semilla=SEMILLA):
    """La fábrica de una red para con_tabla."""
    construir, secuencia = CANDIDATAS[nombre]
    return lambda: Red(construir, secuencia, epocas, semilla)


def repetir_con_semillas(serie, nombre, epocas):
    """Repite la mejor red con cinco semillas en el backtest completo."""
    semillas = [2026, 2027, 2028, 2029, 2030]
    filas = []
    for s in semillas:
        _, promedio = backtest(serie, con_tabla(red(nombre, epocas, s)))
        filas.append({
            "modelo": nombre,
            "epocas": epocas,
            "semilla": s,
            "RMSE": round(float(promedio["RMSE"]), 2),
            "MAE": round(float(promedio["MAE"]), 2),
            "MAPE": round(float(promedio["MAPE"]), 2),
        })
        print(f"  Semilla {s}: RMSE={promedio['RMSE']:.2f}, MAE={promedio['MAE']:.2f}, MAPE={promedio['MAPE']:.2f}%")

    tabla_s = pd.DataFrame(filas)
    tabla_s.to_csv(RESULTADOS / "semillas.csv", index=False)
    media = tabla_s["RMSE"].mean()
    desv = tabla_s["RMSE"].std(ddof=1)
    print(f"  Estabilidad {nombre} (5 semillas, ddof=1): media RMSE={media:.2f}, desviación={desv:.2f}")


# ============================================================
# 5. La tabla y la elección
# ============================================================

def contar_parametros(ajustado):
    """Cuántos números guarda el modelo ajustado. En los clásicos y las redes
    son los parámetros; en KNN, los valores que memoriza; en SVR, los vectores
    de soporte y sus coeficientes; en los árboles, los nodos."""
    if hasattr(ajustado, "params"):
        return len(ajustado.params)
    if isinstance(ajustado, Red):
        return sum(p.numel() for p in ajustado.red.parameters())
    final = ajustado.steps[-1][1] if hasattr(ajustado, "steps") else ajustado
    if isinstance(final, KNeighborsRegressor):
        return final.n_samples_fit_ * final.n_features_in_
    if isinstance(final, SVR):
        return final.support_vectors_.size + final.dual_coef_.size + 1
    if isinstance(final, RandomForestRegressor):
        return sum(arbol.tree_.node_count for arbol in final.estimators_)
    if isinstance(final, XGBRegressor):
        return len(final.get_booster().trees_to_dataframe())
    return 0


def fila(nombre, serie, modelo):
    ULTIMO.clear()
    inicio = time.perf_counter()
    _, promedio = backtest(serie, modelo)
    segundos = time.perf_counter() - inicio
    parametros = contar_parametros(ULTIMO["modelo"]) if "modelo" in ULTIMO else 0
    print(f"  {nombre}: RMSE {promedio['RMSE']:.2f} en {segundos:.1f} s")
    return {"modelo": nombre, "RMSE": round(promedio["RMSE"], 2),
            "MAE": round(promedio["MAE"], 2), "MAPE": round(promedio["MAPE"], 2),
            "parametros": parametros, "segundos": round(segundos, 1)}


def graficar_pronosticos(serie, modelos):
    """Pronóstico desde el último origen del backtest de cada modelo elegido."""
    entrenamiento, real = serie.iloc[:-12], serie.iloc[-12:]
    figura, eje = plt.subplots(figsize=(9.5, 3.8))
    eje.plot(serie.index[-36:], serie.iloc[-36:].to_numpy(), color="black",
             linewidth=1.5, label="Real")
    colores = ["#1f4e79", "#d95f02", "#2e6f40", "#7570b3"]
    for idx, (nombre, modelo) in enumerate(modelos.items()):
        pronostico = modelo(entrenamiento, 12)
        eje.plot(real.index, pronostico, marker="o", markersize=3.5, linewidth=1.3,
                 color=colores[idx % len(colores)],
                 label=f"{nombre} (RMSE {rmse(real, pronostico):.1f})")
    eje.axvline(entrenamiento.index[-1], color="gray", linestyle=":", linewidth=1.0)
    eje.set_xlabel("Fecha")
    eje.set_ylabel("Millones de USD")
    eje.set_title("Pronóstico a 12 meses desde el último origen (2025-08)")
    eje.grid(True, alpha=0.25)
    eje.spines["top"].set_visible(False)
    eje.spines["right"].set_visible(False)
    eje.legend(frameon=False, fontsize=8)
    figura.tight_layout()
    figura.savefig(FIGURAS / "pronosticos.png", dpi=150)
    plt.close(figura)


def analizar_errores(serie, modelos):
    """Registra errores por mes objetivo y por horizonte sobre los orígenes del backtest."""
    registros = []
    primero = len(serie) - N_PRUEBA
    h_max = 12
    for nombre, modelo in modelos.items():
        for corte in range(primero, len(serie) - h_max + 1):
            entrenamiento = serie.iloc[:corte]
            reales = serie.iloc[corte:corte + h_max]
            preds = np.asarray(modelo(entrenamiento, h_max))
            for paso, (fecha, real_val, pred_val) in enumerate(
                zip(reales.index, reales.to_numpy(), preds), start=1
            ):
                err = pred_val - real_val
                registros.append({
                    "modelo": nombre,
                    "origen": f"{serie.index[corte - 1]:%Y-%m}",
                    "fecha_objetivo": f"{fecha:%Y-%m}",
                    "mes": fecha.month,
                    "horizonte": paso,
                    "real": round(float(real_val), 2),
                    "pronostico": round(float(pred_val), 2),
                    "error_abs": round(float(abs(err)), 2),
                    "error_cuad": float(err ** 2),
                })
    df = pd.DataFrame(registros)
    por_mes = (
        df.groupby(["modelo", "mes"])
        .agg(RMSE=("error_cuad", lambda s: round(float(np.sqrt(s.mean())), 2)),
             MAE=("error_abs", lambda s: round(float(s.mean()), 2)))
        .reset_index()
    )
    por_mes.to_csv(RESULTADOS / "errores_mes.csv", index=False)

    por_hor = (
        df.groupby(["modelo", "horizonte"])
        .agg(RMSE=("error_cuad", lambda s: round(float(np.sqrt(s.mean())), 2)),
             MAE=("error_abs", lambda s: round(float(s.mean()), 2)))
        .reset_index()
    )
    por_hor.to_csv(RESULTADOS / "errores_horizonte.csv", index=False)

    print("  Resumen de errores por mes (top 3 meses con mayor RMSE):")
    for nombre in modelos:
        sub = por_mes[por_mes["modelo"] == nombre].sort_values("RMSE", ascending=False).head(3)
        resumen_m = ", ".join(f"mes {int(r.mes)} ({r.RMSE:.1f})" for r in sub.itertuples())
        print(f"    {nombre}: {resumen_m}")


def pendiente(parte, error):
    print(f"  pendiente: {parte} ({error})")


def main():
    inicio = time.perf_counter()
    cruda = cargar_serie(ARCHIVO, FECHA, VALOR)
    print(f"Serie: {len(cruda)} meses, de {cruda.index[0]:%Y-%m} a "
          f"{cruda.index[-1]:%Y-%m}, {cruda.isna().sum()} vacíos")
    serie = preparar(cruda)
    graficar_serie(serie)

    print("1. Línea base y modelo a superar")
    filas = [fila("estacional ingenuo", serie,
                  lambda t, h: estacional_ingenuo(t, h, PERIODO))]
    elegidos_para_figura = {}
    try:
        clasico = modelo_clasico(serie)
        filas.append(fila(NOMBRE_CLASICO, serie, clasico))
        elegidos_para_figura[NOMBRE_CLASICO] = clasico
    except NotImplementedError as error:
        pendiente("parte 1", error)

    print("2. Tabla supervisada")
    try:
        comprobar_sin_fuga(serie)
        tabla_lista = True
    except NotImplementedError as error:
        pendiente("parte 2", error)
        tabla_lista = False

    if tabla_lista:
        print("3. Aprendizaje automático")
        try:
            elegidos = ajustar_hiperparametros(serie)
            graficar_importancia(serie, elegidos)
            modelos_aa = {nombre: con_tabla(lambda f=fabrica, p=elegidos[nombre]: f(**p))
                          for nombre, (fabrica, _) in REJILLAS.items()}
            filas_aa = [fila(nombre, serie, modelo) for nombre, modelo in modelos_aa.items()]
            filas += filas_aa
            mejor_aa = min(filas_aa, key=lambda f: f["RMSE"])["modelo"]
            elegidos_para_figura[mejor_aa] = modelos_aa[mejor_aa]
        except NotImplementedError as error:
            pendiente("parte 3", error)

        print("4. Aprendizaje profundo")
        try:
            epocas = validar_redes(serie)
            filas_red = [fila(f"{nombre}, {e} épocas", serie, con_tabla(red(nombre, e)))
                         for nombre, e in epocas.items()]
            filas += filas_red
            mejor = min(zip(epocas, filas_red), key=lambda par: par[1]["RMSE"])[0]
            elegidos_para_figura[mejor] = con_tabla(red(mejor, epocas[mejor]))
            repetir_con_semillas(serie, mejor, epocas[mejor])
        except NotImplementedError as error:
            pendiente("parte 4", error)

    print("5. La tabla")
    tabla = pd.DataFrame(filas).sort_values("RMSE")
    tabla.to_csv(RESULTADOS / "metricas.csv", index=False)
    print(tabla.to_string(index=False))
    print("Mejor modelo por RMSE:", tabla.iloc[0]["modelo"])
    if len(elegidos_para_figura) > 1:
        graficar_pronosticos(serie, elegidos_para_figura)
        analizar_errores(serie, elegidos_para_figura)
    print(f"Tiempo total: {time.perf_counter() - inicio:.0f} s")


if __name__ == "__main__":
    main()
