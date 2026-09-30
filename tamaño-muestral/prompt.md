Quiero montar una pequeña web app, simplemente un HTML con JavaScript, que me permita resolver la pregunta de cuántos data points necesito para saber si una decisión es estadísticamente fuerte o no.

Tengo que entender el approach que hay que seguir aquí:
1. Definir qué input values tengo que tomar.
2. Pintar una serie de gráficas y una distribución.

Primero vamos a definir qué es lo que queremos y luego ya pasamos a la implementación. Básicamente, me quiero enfrentar a una decisión: ¿cuántos puntos hacen falta y cuál es la relación entre esa cantidad de puntos y la fuerza estadística de los mismos?

El enfoque debe ser ilustrativo y fácil de entender para gente que no tenga conocimientos estadísticos potentes, pero siempre con un modo avanzado en el que se muestre la formulación.




Respecto a tu pregunta 1, entiendo que ABT es de media, pero no estoy seguro. La situación que quiero modelar es, por ejemplo: estamos intentando determinar si a las empresas les gusta externalizar. Le hemos preguntado a 140 empresas en España y nos han dicho que no quieren externalizar. Entiendo que eso sería un test de media o clasificación; la métrica paso-umbral quizás sea esa.

Respecto a tu pregunta número 2, el riesgo que más me importa en este caso es el de falso positivo.

Sobre la pregunta 3, ¿qué tamaño de defecto te interesa detectar? No hace falta que haya estacionalidad ni nada parecido, los data points son independientes.

En cuanto a la pregunta 5, prefiero exactitud en lugar de una aproximación normal.

Respecto a la pregunta 6 sobre qué quiero que la app enseñe visualmente, me gusta la opción 1 que propones: las distribuciones. También me encajan "Power versus N" y el resto interactivo.

7. ¿Cuál es el output que esperas?
Necesito que para detectar un efecto de X, con alfa y Power, me indique: "necesitas N por grupo". Sí, esa opción me interesa.