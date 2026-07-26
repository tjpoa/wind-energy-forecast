FROM apache/airflow:3.3.0-python3.11

COPY --chown=airflow:root airflow/requirements-batch.txt /tmp/requirements-batch.txt
RUN pip install --no-cache-dir --requirement /tmp/requirements-batch.txt

COPY --chown=airflow:root pyproject.toml /opt/wind-energy-forecast/pyproject.toml
COPY --chown=airflow:root src /opt/wind-energy-forecast/src
COPY --chown=airflow:root scripts /opt/wind-energy-forecast/scripts
COPY --chown=airflow:root config /opt/wind-energy-forecast/config
COPY --chown=airflow:root airflow/dags /opt/airflow/dags
COPY --chown=airflow:root airflow/tests /opt/wind-energy-forecast/airflow/tests
COPY --chown=airflow:root tests/test_incremental.py /opt/wind-energy-forecast/tests/test_incremental.py
RUN pip install --no-cache-dir --no-deps /opt/wind-energy-forecast

ENV PYTHONPATH=/opt/wind-energy-forecast/src
WORKDIR /opt/wind-energy-forecast
