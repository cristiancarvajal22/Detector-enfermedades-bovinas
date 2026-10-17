"""Red didáctica: 2 entradas → 3 neuronas ocultas → 1 salida sigmoide."""
import numpy as np


def sigmoide(x):
    # Limitar evita overflow numérico cuando la exponencial recibe valores grandes.
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def main():
    entrada = np.array([[0.3, 0.8]])
    pesos_ocultos = np.array([[0.2, -0.4, 0.1], [0.7, 0.5, -0.3]])
    sesgos_ocultos = np.array([0.1, 0.0, -0.1])
    pesos_salida = np.array([[0.6], [-0.2], [0.4]])
    sesgo_salida = np.array([0.05])
    suma_oculta = entrada @ pesos_ocultos + sesgos_ocultos
    activacion_oculta = sigmoide(suma_oculta)
    suma_salida = activacion_oculta @ pesos_salida + sesgo_salida
    salida = sigmoide(suma_salida)
    print("DEMO CON PESOS FIJOS: no es el modelo bovino ni un entrenamiento.")
    for nombre, valor in (("Entrada", entrada), ("W1", pesos_ocultos), ("b1", sesgos_ocultos),
                          ("z1 = xW1 + b1", suma_oculta), ("a1 = sigmoide(z1)", activacion_oculta),
                          ("W2", pesos_salida), ("b2", sesgo_salida), ("z2 = a1W2 + b2", suma_salida),
                          ("Salida = sigmoide(z2)", salida)):
        print(f"{nombre}:\n{valor}")


if __name__ == "__main__":
    main()
