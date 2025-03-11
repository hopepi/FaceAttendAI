import os
import shutil
from PIL import Image

def veri_setini_farkli_yere_olustur(kaynak_dizin, hedef_dizin):

    desteklenen_formatlar = ['.jpg', '.jpeg', '.png']

    dosyalar = os.listdir(kaynak_dizin)

    for dosya in dosyalar:

        dosya_yolu = os.path.join(kaynak_dizin, dosya)

        dosya_uzantisi = os.path.splitext(dosya)[1].lower()

        if os.path.isfile(dosya_yolu) and dosya_uzantisi in desteklenen_formatlar:

            klasor_ismi = os.path.splitext(dosya)[0]
            klasor_yolu = os.path.join(hedef_dizin, klasor_ismi)

            if not os.path.exists(klasor_yolu):
                os.makedirs(klasor_yolu)

            yeni_dosya_yolu = os.path.join(klasor_yolu, dosya)

            with Image.open(dosya_yolu) as img:
                img = img.resize((160, 160))
                img.save(yeni_dosya_yolu)

            print(f"{dosya} dosyası 160x160 boyutuna dönüştürüldü ve {klasor_yolu} klasörüne kopyalandı.")

kaynak_dizin = r"C:\Users\ingin\OneDrive\Masaüstü\1"
hedef_dizin = r"C:\Users\ingin\OneDrive\Belgeler\Sınıf dataset"

if not os.path.exists(hedef_dizin):
    os.makedirs(hedef_dizin)

veri_setini_farkli_yere_olustur(kaynak_dizin, hedef_dizin)