from rn.models import Gallery
from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
import base64
from django.conf import settings
import os
from django.core.files.base import ContentFile
from rn.utils.image_detect import ImageProccess
# Create your views here.


def base64ToImage(img_data):
    return ContentFile(base64.b64decode(img_data))


@api_view(["POST"])
def ReconizeObjects(request):
    if request.method == "POST":
        image_data = ""
        tag = ""

        try:
            image_data = request.data["image_data"]
            tag = request.data["tag"]
        except:
            return Response(status=status.HTTP_400_BAD_REQUEST)

        IMAGE = base64ToImage(image_data)
        gallery = Gallery()
        gallery.tag = tag
        gallery.save()
        #
        IMAGE_NAME = tag
        gallery_final = Gallery.objects.get(id=gallery.id)
        gallery_final.image.save(IMAGE_NAME, IMAGE, save=True)
        #
        final_image, count_items = ImageProccess(IMAGE_NAME)
        return Response(status=status.HTTP_200_OK, data={
            "image_data": final_image,
            "count_items": count_items
        })
