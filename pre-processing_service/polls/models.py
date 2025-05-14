from django.db import models

class Category(models.Model):
    identifier = models.AutoField(primary_key=True)   # Автоинкрементный идентификатор
    name = models.CharField(max_length=255, verbose_name="Название категории")
    emoji = models.CharField(max_length=10, verbose_name="Эмодзи", default='')

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = 'Категория'
        verbose_name_plural = 'Категории'

class Subcategory(models.Model):
    identifier = models.AutoField(primary_key=True)   # Автоинкрементный идентификатор
    name = models.CharField(max_length=255, verbose_name="Название подкатегории")
    description = models.TextField(verbose_name="Описание подкатегории")
    category = models.ForeignKey(Category, related_name='subcategories', on_delete=models.CASCADE)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = 'Подкатегория'
        verbose_name_plural = 'Подкатегории'
        
class Up_Fle(models.Model):
    file = models.FileField(upload_to='uploads/originals/%Y/%m/%d/', verbose_name='Загруженный файл')
    processed_file = models.FileField(
        upload_to='uploads/processed/%Y/%m/%d/',
        null=True,
        blank=True,
        verbose_name='Обработанный файл'
    )
    description = models.TextField(blank=True, null=True, verbose_name='Описание')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Файл от {self.uploaded_at.strftime('%d.%m.%Y %H:%M')}"

    class Meta:
        verbose_name = 'Загруженный файл'
        verbose_name_plural = 'Загруженные файлы'