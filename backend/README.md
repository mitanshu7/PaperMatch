# TODO
- [ ] Redis for caching

Build with docker:

docker build -t papermatch_backend .
docker run --env-file .env  -p 8001:8001 papermatch_backend:latest 
