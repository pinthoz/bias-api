FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r requirements.txt

# int8 model (~105 MB): loads within Lambda's 10 s init window; the fp32
# model.onnx is left out by .dockerignore (see export/quant_results.json).
# Its thresholds were re-tuned on the validation split with probabilities
# computed on the Lambda's own arm64 CPU (export/retune_results.arm64.json):
# int8 outputs differ between x86 and ARM
COPY model/ /opt/model/
ENV MODEL_FILE=model.int8.onnx THRESHOLDS_FILE=thresholds.int8.arm64.json
COPY app.py ${LAMBDA_TASK_ROOT}

CMD ["app.handler"]