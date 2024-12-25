FROM python:3.9
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN pip install -U "huggingface_hub[cli]" ultralytics scikit-image
RUN git config --global credential.helper store && huggingface-cli login --token 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 --add-to-git-credential
RUN huggingface-cli download smartsurgery/dentistry-models all_category/dentistry_yolov11x-seg-all_4.42.pt --local-dir /tmp && \
huggingface-cli download smartsurgery/dentistry-models dental_contour/dentistryContour_yolov11n-seg_4.46.pt --local-dir /tmp
RUN mkdir -p /opt/nuclio /opt/nuclio/tests &&\
mv /tmp/all_category/dentistry_yolov11x-seg-all_4.42.pt /opt/nuclio/segmentation-model.pt && \
mv /tmp/dental_contour/dentistryContour_yolov11n-seg_4.46.pt /opt/nuclio/contour-model.pt && \
rm /tmp/* -rf
COPY . /opt/nuclio/
RUN curl -o /opt/nuclio/tests/test.jpeg https://i.imgur.com/Dsd1kUY.jpeg
CMD [ "python", "/opt/nuclio/main.py" ]