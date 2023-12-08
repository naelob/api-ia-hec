# api-ia-hec

# Deployed on ec2 instance on AWS

<http://ec2-54-204-174-230.compute-1.amazonaws.com>

## Build Docker Image

```
docker build -t main .
```

## Launch Docker Image

```
docker run -d -p 8000:8000 main
```

The API should be exposed at <http://0.0.0.0:8000>
