# ¿Me puedo fiar de mis datos? — especificación

App web en un único HTML autónomo (vanilla JS + SVG/Canvas, fuentes y KaTeX por CDN). Todo corre en el navegador. Estética lúdica/visual. Público: gente sin formación estadística, con un modo avanzado que muestra la formulación.

## Flujo

1. **¿Qué es cada dato?** — Sí/no · Un número · Un conteo de sucesos · Una categoría entre varias.
2. **¿Qué quieres saber?** — Estimar · Frente a un umbral · Comparar dos grupos · Antes/después (mismos sujetos).
3. **Tus datos** — formulario específico de cada caso («cuatro datos»), relleno con un ejemplo.
4. **Veredicto** — semáforo + frase en lenguaje llano + rango plausible + fragilidad.
5. **Repite tu estudio** — simulación de N repeticiones (configurable, 10–10.000) con interruptor de «verdad».
6. **¿Cuántos datos necesitas?** — N para estar en verde + deslizador de N.
7. **Modo avanzado** (interruptor global) — test usado, fórmulas (KaTeX), p-valor, números del cálculo.

## Tests (política: exacto cuando existe; si no, Monte Carlo/permutación; nunca aproximación normal salvo como comparativa)

| | Estimar | Umbral | Dos grupos | Antes/después |
|---|---|---|---|---|
| Sí/no | Clopper-Pearson | Binomial exacto | Fisher exacto | McNemar exacto |
| Número | Bootstrap (datos) / t (resumen) | Permutación de signos / t | Permutación / Welch | Permutación de signos / t pareado |
| Conteo | Poisson exacto (Garwood) | Poisson exacto | Test condicional de tasas | Test condicional de tasas |
| Categorías | IC Clopper-Pearson simultáneos (Bonferroni) | ¿Ganador claro? (1.ª vs 2.ª, binomial exacto) | χ² Monte Carlo con márgenes fijos (Freeman-Halton) | Bowker Monte Carlo |

- Números: tres formas de entrada — pegar datos, media + desviación típica, o rango típico (DE ≈ (máx − mín)/4).
- En la simulación, cada repetición se decide con la versión rápida del test cuando el exacto requeriría Monte Carlo anidado (t, χ²). Se indica en modo avanzado.

## Semáforo

- **Estimar**: el usuario fija el margen aceptable. Verde si margen ≤ objetivo; ámbar si ≤ 2× objetivo; rojo si mayor.
- **Tests**: índice de fragilidad.
  - Sí/no y categorías (ganador): nº de respuestas que habría que cambiar para que cambie la conclusión.
  - Número y conteo: nº de valores/sucesos extremos que habría que quitar (o añadir) para que cambie.
  - Verde: significativo y fragilidad > límite. Ámbar: significativo pero frágil, o no significativo por poco. Rojo: no significativo y lejos.
  - Límite de fragilidad: max(2, ⌈2 % de n⌉).
  - Donde la fragilidad no aplica (resúmenes numéricos, categorías × grupos, pareado de categorías): ámbar si α/2 < p ≤ 2α.

## Simulación

- Nº de repeticiones configurable (10–10.000).
- Modo «si lo que viste es la verdad»: cuántas repeticiones detectan el efecto (potencia bajo lo observado).
- Modo «si no hubiera efecto»: cuántas repeticiones te engañarían (falsos positivos ≈ α).
- En «Estimar»: cuántas repeticiones caen dentro del margen aceptable.
- Rejilla de puntos (canvas) + histograma de estimaciones.

## Planificación

- Dentro del resultado: «para estar en verde necesitarías N» + deslizador de N con potencia (o margen) frente a N.
- Potencia exacta en sí/no (umbral y dos grupos); Monte Carlo en el resto.
