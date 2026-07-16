# SimDrive — İlerleme Takibi

Kutucukları ben (Recephan) işaretliyorum. Kural: bir maddeyi ancak o aracı
kendim çalıştırıp sonucunu görüp "neden böyle oldu"yu anlatabiliyorsam işaretlerim.
Kod yazmak yok — kodu AI yazar; ben sistemi kurar, ayarlar, ölçer ve yönetirim.

## Hafta 1 — Tespit (araç: YOLOv8 / ultralytics)
- [ ] detect_video.py'yi kendi videomla çalıştır
- [ ] --conf eşiğiyle oyna (0.1 vs 0.6): düşük eşik ne getirir, ne götürür? (precision/recall mantığı)
- [ ] --model yolov8s.pt ile dene: hız vs doğruluk takasını kendi gözümle gör
- [ ] "YOLO tek bakışta nasıl tespit yapıyor?" — mantığını 3-4 cümleyle anlatabilir ol

## Hafta 2 — Takip + Şerit (araç: ByteTrack, OpenCV)
- [ ] track_video.py'yi çalıştır; ID'lerin ne işe yaradığını gör
- [ ] "Tespit varken takip niye ayrıca lazım?" sorusunu cevaplayabil
- [ ] Dashcam açılı video bul, lane_detect.py'yi üzerinde çalıştır (--debug ile)
- [ ] ROI/Canny parametrelerini kendi videoma göre ayarla (Claude'a hangi değerlerin işe yaradığını raporla)
- [ ] "Neden şerit için derin öğrenme değil de klasik yöntem?" — artı/eksileri söyleyebil

## Hafta 3 — Mesafe + Uyarı + Arayüz (araç: Streamlit)
- [ ] Kutu boyutundan mesafe tahmini mantığını anla (neden kamera kalibrasyonu önemli?)
- [ ] Uyarı eşiklerine BEN karar vereyim (kaç metrede uyarsın? kaç saniyede?)
- [ ] Streamlit uygulamasını çalıştır, kullan, "kullanıcı olarak" eksik bul ve iyileştirme iste
- [ ] Demo videosu kaydet (portfolyo malzemesi)

## Hafta 4 — Unity Perception (benim saham!)
- [ ] Unity Perception paketini kur, örnek sahneyi çalıştır
- [ ] Trafik sahnesini BEN kurayım (asset seçimi, sahne düzeni — Unity bilgim burada konuşur)
- [ ] Domain randomization ayarlarına BEN karar vereyim (ışık/renk/açı aralıkları)
- [ ] YOLO formatında etiketli sentetik dataset üret ve kalitesini kontrol et

## Hafta 5 — Fine-tune (araç: ultralytics train)
- [ ] Eğitimi kendim başlat (data.yaml, epochs, imgsz — parametrelere ben karar vereyim)
- [ ] Eğitim çıktısını okumayı öğren: loss eğrileri, mAP50, mAP50-95 ne anlatır?
- [ ] "Overfitting sentetik veride nasıl görünür?" sorusunu cevaplayabil

## Hafta 6 — Deney (asıl olay)
- [ ] Deney tasarımına BEN karar vereyim: hangi videolarda, hangi metrikle karşılaştıracağız?
- [ ] Karşılaştırmayı çalıştır, sonuç tablosunu yorumla
- [ ] experiments/ altında sonuç raporu: "sim-to-real transfer işe yaradı mı?"
- [ ] README'ye sonuçları ekle (portfolyonun vitrini)

## Devlog (docs/devlog/gun-NN.md)
- [ ] Gün 1 yazıldı
<!-- her çalışma günü için bir satır ekle; commit'i kendim atarım -->
