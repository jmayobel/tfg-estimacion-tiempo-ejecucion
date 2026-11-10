# TFG-Benchmark

Código del Trabajo Fin de Grado *Estimación del tiempo de ejecución de cargas de trabajo en
arquitecturas heterogéneas*.

El sistema ejecuta un catálogo de servicios en contenedores Docker en varias máquinas, mide su
tiempo de ejecución y el uso de recursos, y construye con ello un dataset. Sobre ese dataset se
entrenan y comparan modelos que predicen el tiempo de ejecución en máquinas que no se han usado
para entrenar.

## Estructura

| Ruta | Contenido |
|------|-----------|
| `scripts/orchestrator.sh` | Orquestador de la recogida de datos en una máquina |
| `scripts/run_scaled_batch.sh`, `scripts/run_replicas.sh` | Lanzan los cuatro tamaños de carga y sus réplicas |
| `scripts/scales/` | Tamaño de cada carga (`small`, `medium`, `large`, `xlarge`) |
| `scripts/hw_profile.py`, `scripts/calibrate.py` | Características y calibración de la máquina |
| `docker-compose.services.yml` | Catálogo de servicios (un perfil de Compose por servicio) |
| `docker-compose.monitoring.yml`, `prometheus/` | Monitorización (Prometheus, cAdvisor y node exporter) |
| `scripts/export_csv.py` | Exportación de las métricas de cada servicio a CSV |
| `scripts/build_ml_dataset.py`, `scripts/service_profiles.yaml` | Agregación de cada ejecución en una fila del dataset |
| `scripts/train_baseline.py` | Limpieza del dataset, modelos y validación |
| `scripts/ab_workload_scale.py`, `scripts/bootstrap_exponents.py`, `scripts/model_diagnostics.py` | Análisis adicionales de los resultados |
| `scripts/error_por_servicio.py` | Error de cada servicio con máquinas conocidas |
| `scripts/loso_por_servicio.py`, `scripts/vecinos_clase.py`, `scripts/loso_maquina_servicio.py`, `scripts/loso_tamano.py`, `scripts/bootstrap_ram.py` | Software no conocido, tamaños de carga no medidos e intervalos del exponente de la memoria |
| `datasets/<máquina>/` | Datos de las ocho máquinas (`runs/`, `ml_dataset.csv` y, si hace falta, `host_overrides.json`) |
| `azure/`, `scripts/systemd/` | Despliegue de las máquinas virtuales de Azure (Bicep) y reanudación automática |
| `scripts/collect_remote.sh`, `scripts/download_datasets.sh`, `scripts/remote_hosts.txt` | Recogida de datos en máquinas remotas y descarga de sus resultados |
| `scripts/backfill_calibration.py`, `scripts/backfill_hw_profile.py`, `scripts/recalibrate_local.sh`, `scripts/merge_datasets.py`, `scripts/fewshot_fingerprint.py` | Scripts auxiliares y exploratorios que no forman parte del anexo |
| `tests/` | Pruebas automáticas |
| `E-VRPTW Instances/` | Instancia de entrada del servicio `busqueda-caminos` y descripción de su formato |

## Reproducir los resultados

No hace falta volver a medir, porque los datos de las ocho máquinas están en `datasets/`. Se
necesita Python 3.11 o posterior. Las versiones de `requirements-resultados.txt` son las mismas con
las que se obtuvieron las cifras de la memoria.

```bash
pip install -r requirements-resultados.txt
python3 scripts/train_baseline.py | tee output/train_results.txt
python3 scripts/ab_workload_scale.py
python3 scripts/bootstrap_exponents.py
python3 scripts/model_diagnostics.py | tee output/diagnostics_results.txt
```

`train_baseline.py` carga las 889 ejecuciones, la limpieza deja 631 y, en el esquema que aparta una
máquina entera (`GroupKFold por HOST`), el modelo log-lineal con calibración (`LogLinear_Svc_Full`)
obtiene un error porcentual mediano del 24,4 % y un R² de 0,858.

El error de cada servicio con máquinas conocidas, los resultados con software no conocido, con
tamaños de carga no medidos y la explicación de los intervalos del exponente de la memoria se
obtienen con estos scripts:

```bash
python3 scripts/error_por_servicio.py
python3 scripts/loso_por_servicio.py
python3 scripts/vecinos_clase.py
python3 scripts/loso_maquina_servicio.py
python3 scripts/loso_tamano.py
python3 scripts/bootstrap_ram.py
```

Todos guardan sus resultados en `output/`. En Windows conviene definir `PYTHONIOENCODING=utf-8`
para que las tildes de la salida se vean bien.

Para regenerar el dataset de una máquina a partir de sus ejecuciones:

```bash
python3 scripts/build_ml_dataset.py \
  --runs-dir datasets/<máquina>/runs \
  --output datasets/<máquina>/ml_dataset.csv
```

## Recoger datos nuevos

Se necesita Docker con el complemento Compose. En el fichero `.env` se indica el nombre de la
máquina (`HOSTNAME_LABEL`) y el tamaño de las cargas. Después:

```bash
make run-docker                 # una pasada de todos los servicios
scripts/run_scaled_batch.sh     # los cuatro tamaños de carga, uno tras otro
```

Cada pasada crea una carpeta en `output/runs/` con un CSV por servicio, `hw_profile.json`,
`run_metadata.json` y el registro del orquestador. Las guías de instalación por plataforma están
en `README-WINDOWS.md`, `README-MACOS.md` y `README-AZURE.md`.

## Pruebas

```bash
make install-dev
make test
```
