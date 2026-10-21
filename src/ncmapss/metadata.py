"""Metadatos del dataset (nombres y significado de las variables, según Arias Chao et al., 2021).

No son resultados: describen qué mide cada columna. Todo número del análisis se calcula
desde los datos en eda.py y en los módulos de modelado.
"""

DESCRIPTIONS = {
    "alt": "Altitud [ft]", "Mach": "Número de Mach", "TRA": "Ángulo de palanca de empuje [%]",
    "T2": "Temperatura total a la entrada del fan [°R]",
    "T24": "Temp. total a la salida del LPC [°R]", "T30": "Temp. total a la salida del HPC [°R]",
    "T48": "Temp. total a la salida de la HPT [°R]", "T50": "Temp. total a la salida de la LPT [°R]",
    "P15": "Presión total en el bypass [psia]", "P2": "Presión total a la entrada del fan [psia]",
    "P21": "Presión total a la salida del fan [psia]", "P24": "Presión total a la salida del LPC [psia]",
    "Ps30": "Presión estática a la salida del HPC [psia]", "P40": "Presión total a la salida del combustor [psia]",
    "P50": "Presión total a la salida de la LPT [psia]", "Nf": "Velocidad física del fan [rpm]",
    "Nc": "Velocidad física del núcleo [rpm]", "Wf": "Flujo de combustible [pps]",
    "T40": "Temp. total a la salida del combustor [°R]", "P30": "Presión total a la salida del HPC [psia]",
    "P45": "Presión total a la salida de la HPT [psia]", "W21": "Flujo a la salida del fan [pps]",
    "W22": "Flujo a la salida del LPC [pps]", "W25": "Flujo a la entrada del HPC [pps]",
    "W31": "Flujo de refrigeración HPT [pps]", "W32": "Flujo de refrigeración HPT [pps]",
    "W48": "Flujo a la salida de la HPT [pps]", "W50": "Flujo a la salida de la LPT [pps]",
    "SmFan": "Margen de stall del fan [-]", "SmLPC": "Margen de stall del LPC [-]",
    "SmHPC": "Margen de stall del HPC [-]", "phi": "Relación combustible / Ps30 [pps/psi]",
    "fan_eff_mod": "Modificador de eficiencia del fan", "fan_flow_mod": "Modificador de flujo del fan",
    "LPC_eff_mod": "Mod. eficiencia LPC", "LPC_flow_mod": "Mod. flujo LPC",
    "HPC_eff_mod": "Mod. eficiencia HPC", "HPC_flow_mod": "Mod. flujo HPC",
    "HPT_eff_mod": "Mod. eficiencia HPT", "HPT_flow_mod": "Mod. flujo HPT",
    "LPT_eff_mod": "Mod. eficiencia LPT", "LPT_flow_mod": "Mod. flujo LPT",
    "RUL": "Vida útil remanente [ciclos]", "unit": "Motor", "cycle": "Ciclo (vuelo)",
    "Fc": "Clase de vuelo (1 corto, 2 medio, 3 largo)", "hs": "Estado de salud (1 sano, 0 degradado)",
}

BLOCKS = {
    "W": ("Condiciones de vuelo", "Altitud, velocidad, potencia pedida y temperatura del aire que entra."),
    "X_s": ("Sensores físicos", "Lo que mide un motor real: temperaturas, presiones, giros y combustible."),
    "X_v": ("Sensores virtuales", "Valores internos del simulador. No existen en un avión real."),
    "T": ("Salud (θ)", "Cuánto se desgastó cada componente. Tampoco se mide en la realidad."),
    "Y": ("Objetivo", "Vuelos que le quedan al motor antes de fallar."),
    "A": ("Auxiliares", "Motor, número de vuelo, clase de vuelo y estado de salud."),
}

COMPONENTS = ["fan", "LPC", "HPC", "HPT", "LPT"]

SHORT_LABEL = {"alt": "Altitud [ft]", "Mach": "Velocidad [Mach]", "TRA": "Palanca de potencia, TRA [%]",
               "T2": "Temperatura del aire de entrada, T2 [°R]", "T48": "Temperatura de turbina, T48 [°R]"}


def component_of(param: str) -> str:
    """'HPT_eff_mod' -> 'HPT'."""
    return param.split("_")[0]


def short_param(param: str) -> str:
    """'HPT_eff_mod' -> 'HPT_eff'."""
    return param.replace("_mod", "")
