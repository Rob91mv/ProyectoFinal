# PROYECTO PARA INTERCONNECT
#TTPF

## Objetivo del Proyecto Final - Bootcamp Data Science.

El objetivo de este proyecto es asesorar a la compañía de telecomunicaciones Interconnect sobre la creación de un modelo predictivo de la tasa de cancelación de clientes, de tal forma de que la compañía pueda anticiparse para ofrecer códigos promocionlaes y opciones de planes especiales. 

Para ello se utilizaran bases de datos facilitadas por el equipo de marketing de Interconnect que recopila datos desde el 2016 en adelante.

Las métricas clave de este proyecto serán: calidad del modelo a través de la métrica AUC-ROC y la "exactitud".

## Datasets facilitados

* contract : contiene información sobre los contratos de los usuarios de Interconnect (fecha inicio y fin de contrato, modo de pago, tipo de contrato, etc)
* personal = contiene información personal de usuario de Interconnect (género, dependientes económicos, seniors, etc.)
* internet = contiene información del servicio de internet y servicios extras (seguridad en linea, streaming, etc.)
* phone = contiene información del servicio telefónico (multiples líneas)

## Meta primordial del proyecto

Predecir la tasa de cancelación de contratos en Interconnect. 

## Alcance

El modelo predictivo a construir permitirá predecir si un cliente cancelará su contrato o no.

## Entregables

* Gráficos EDA: histogramas de variables categóricas, graficos de lineas de variables numericas, matriz de correlacion con variable objetivo, entre otros.
* Archivos .csv que contiene el dataset con datos limpios y dataset listo para el modelado.
* Gráfico de la curva AUC-ROC del modelo con mayor calidad.
* Archivo .csv que contiene la información del mejor modelo.

## Criterios de éxito del proyecto

El proyecto debe tener un valor AUC-ROC mayor o igual a 0.75 en el conjunto de prueba.

## Instrucciones de uso

Ejecutar archivo "Proyecto-Interconnect.py"
Archivo "Proyecto-Interconnect.ipynb" contiene el informe del proyecto.




