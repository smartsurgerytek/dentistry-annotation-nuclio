
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

### Deploy Function to Nuclio
```bash
nuctl deploy --project-name cvat --path "." --platform local
```


### Create a Project and Add Labels
- [Create a Task] and name `Dentistry` and leave the project blank.
- Click `Construction > From Model`.
- Select `Yolo v11x` from the dropdown.
- Click on all the labels to add to the project.
- Click `Done`.
- Select files (haoran will write the file path)
- Click `Submit & Open`.
- Open [Jobs] and click the job.
- Click ![ai-tools-image] from the side bar.
- Click `Detectors` tab.
- Select `Yolo v11x` model from the dropdown.
- Click `Annotate` button.
- Wait a second you should see the image has been auto annotated.



[Serverless Tutorial]: https://docs.cvat.ai/docs/manual/advanced/serverless-tutorial/
[Nuclio CLI]: https://docs.nuclio.io/en/stable/reference/nuctl/nuctl.html
[Create a Project]: http://localhost:8080/projects/create
[Create a Task]: http://localhost:8080/tasks/create
[Jobs]: http://localhost:8080/jobs