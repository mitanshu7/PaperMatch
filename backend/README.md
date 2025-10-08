# TODO
- [ ] Redis for caching

Run with docker:

```bash
docker build -t papermatch_backend .
docker run -d --env-file .env -p 8001:8001 papermatch_backend:latest 
```

Run with podman(quadlets):

```bash
podman build --tag papermatch_backend .

# Create folder for quadlet service
mkdir -p ~/.config/containers/systemd/
nano ~/.config/containers/systemd/papermatch_backend.container
```

write the following:

```ini
[Container]
Image=localhost/papermatch_backend:latest
AutoUpdate=local
PublishPort=8001:8001

EnvironmentFile=/absolute/path/to/.env

[Service]
Restart=always

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user start papermatch_backend.service
```