import io
import json
import logging
from pathlib import Path

import kraken.binarization
import kraken.repo
import kraken.pageseg
from PIL import Image

from .grpc import mocri_pb2 as pb
from .grpc import mocri_pb2_grpc as grpc

APP_DIR = Path('~/.config/kraken').expanduser()

logger = logging.getLogger('mocri')


class MocriService(grpc.MocriServicer):
    @classmethod
    def add_to_server(cls, server):
        grpc.add_MocriServicer_to_server(cls(), server)

    def __init__(self):
        cached_path = Path('/tmp/kraken_models.json')
        if not cached_path.exists():
            logger.info('Fetching list of available models')
            self._models = kraken.repo.get_listing(lambda: None)
        else:
            with cached_path.open('rt') as fp:
                self._models = json.load(fp)

    def _locate_model(self, name):
        for suffix in ('pronn', 'clstm'):
            model_path = APP_DIR / '{}.{}'.format(name, suffix)
            if model_path.exists():
                return model_path
        logger.info("Downloading model '{}'".format(name))
        kraken.repo.get_model(name, str(APP_DIR), lambda: None)
        return self._locate_model(name)

    def ListModels(self, request, context):
        for name, model in self._models.items():
            yield pb.OcrModelInfo(
                name=name,
                description=model.get('summary'),
                scripts=model.get('script', []),
                graphemes=model.get('graphemes', []))

    def BinarizeImage(self, params, context):
        img = Image.open(io.BytesIO(params.image.data))
        binarized = kraken.binarization.nlbin(
            img, threshold=params.threshold, zoom=params.zoom,
            escale=params.escale, border=params.border, perc=params.perc,
            low=params.low, high=params.high)
        out_fp = io.BytesIO()
        binarized.save(out_fp, 'PNG')
        return pb.Image(data=out_fp.getvalue(), mimeType='image/png')

    def SegmentLines(self, params, context):
        img = Image.open(io.BytesIO(params.image.data))
        direction = pb.Direction.Name(
            params.direction).lower().replace('_', '-')
        segments = kraken.pageseg.segment(
            img, text_direction=direction, scale=params.scale,
            maxcolseps=params.maxColSeps, black_colseps=params.blackColSeps)
        for x1, y1, x2, y2 in segments['boxes']:
            yield pb.Box(offsetX=x1, offsetY=y1, width=x2-x1, height=y2-y1)

    def RecognizeText(self, params, context):
        img = Image.open(io.BytesIO(params.image.data))
        model_path = self._locate_model(params.modelName)
        recognizer = kraken.lib.models.load_any(bytes(model_path))
        bounds = {
            'boxes': [
                (l.offsetX, l.offsetY,
                 l.offsetX + l.width, l.offsetY + l.height)
                for l in params.lines],
            'text_direction': pb.Direction.Name(
                params.direction).lower().replace('_', '-')}
        results = kraken.rpred.rpred(
            recognizer, img, bounds, params.padding, params.normalize,
            params.bidiReorder)
        for ocr_record in results:
            yield pb.OcrText(
                text=ocr_record.prediction,
                positions=[
                    pb.Box(offsetX=x, offsetY=y, width=x2-x, height=y2-y)
                    for x, y, x2, y2 in ocr_record.cuts],
                confidences=ocr_record.confidences)
