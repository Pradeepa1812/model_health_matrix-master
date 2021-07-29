# work around --network since mysql cannot be connected to host via ip.
# change the bind address of mysql to allow connections via ip
# then use ip to connect in config.py and remove --network

docker run --name model-health-matrix-app -dit \
--network=host \
-p 9000:9000 \
--restart unless-stopped \
--env APP_SETTINGS=config.Base \
model_health_matrix \
-c gunicorn_config.py \
-b 0.0.0.0:9000 \
--log-level DEBUG \
run:app
