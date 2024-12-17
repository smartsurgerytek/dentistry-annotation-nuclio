
### Install Nuclio

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml -f components/serverless/docker-compose.serverless.yml up -d --build
```
See more from [Serverless Tutorial].

### Install Nuclio Cli
```bash
curl -s https://api.github.com/repos/nuclio/nuclio/releases/latest \
			| grep -i "browser_download_url.*nuctl.*$(uname)" \
			| cut -d : -f 2,3 \
			| tr -d \" \
			| wget -O nuctl -qi - && chmod +x nuctl
```
See more from [Nuclio CLI].

### Deploy function to Nuclio
```bash
nuctl deploy --project-name cvat --path "." --platform local
```

[Serverless Tutorial]: https://docs.cvat.ai/docs/manual/advanced/serverless-tutorial/
[Nuclio CLI]: https://docs.nuclio.io/en/stable/reference/nuctl/nuctl.html