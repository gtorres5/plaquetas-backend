from django.db import models

# Create your models here.


class Gallery(models.Model):
    tag = models.TextField(null=True, blank=True)
    image = models.ImageField(null=True, blank=True)

    class Meta:
        db_table = "gallery"

    def __str__(self):
        return f"{self.id}. {self.tag}"
