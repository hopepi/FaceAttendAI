import cv2


def yatay_cevir_opencv(resim_yolu, kaydet_yolu):
    # Resmi oku
    resim = cv2.imread(resim_yolu)

    # Resmi yatay olarak çevir (1: yatay çevirme, 0: dikey çevirme, -1: her iki yönde çevirme)
    yatay_cevrilmis_resim = cv2.flip(resim, 1)

    # Çevrilmiş resmi kaydet
    cv2.imwrite(kaydet_yolu, yatay_cevrilmis_resim)
    print(f"Resim yatay olarak çevrildi ve {kaydet_yolu} yoluna kaydedildi.")


# Örnek kullanım
resim_yolu = r"../efefoto.jpeg"  # Yatay çevrilecek resmin yolu
kaydet_yolu = "yatay_cevrilmis_resim.jpg"  # Çevrilmiş resmin kaydedileceği yol

yatay_cevir_opencv(resim_yolu, kaydet_yolu)