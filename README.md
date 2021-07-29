# Sandman Model Health Matrix

sandman super admin model health matrix report

### Docker build command:
docker build -t model_health_matrix .

### Docker run command:
docker run --name model-health-matrix-app -d  -p 9000:9000 --restart unless-stopped --env APP_SETTINGS=config.Base model_health_matrix -c gunicorn_config.py -b 0.0.0.0:9000 --log-level DEBUG run:app