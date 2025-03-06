Mevcut model, WIDER FACE veri setiyle eğitilmiş olup yüz algılama konusunda güçlü
bir temel kazanmıştır. Modelin ilk 10 katmanı dondurularak, ekstra 5000 yakın çekim
yüz fotoğrafıyla fine-tuning yapılması hedeflenmiştir. Ancak, uzak mesafedeki yüzleri
tespit etme konusunda beklenen iyileşmeyi sağlamamış, aksine modelin genel performansında
düşüş gözlemlenmiştir. Bu nedenle, bu strateji şimdilik rafa kaldırılmıştır.

Bunun yerine, başlangıçta YOLOv8 kullanılması planlanmışken, yeni strateji olarak
YOLOv11 small modeline geçiş yapılacaktır.

Bu süreci revize etmek isteyenler için bu FineTuning paketi oluşturulmuştur
isteyen kendi projelerinde deneyebilir.
İlgilenenler aşağıdaki bağlantıdan ulaşabilir Akanametov model eğitimi için teşekkürler:
https://github.com/akanametov/yolo-face/tree/dev

Son olarak, modelin teorik olarak düşük doğruluk oranları gözükse de
pratikte birçok alternatif algoritmaya göre daha güçlü olduğu tespit edilmiştir.