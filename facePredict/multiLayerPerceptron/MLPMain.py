import Train
import Test
import Predict

def main():
    print("--- Multi-Layer Perceptron (MLP) Yüz Tanıma ---")
    print("1: Modeli Eğit")
    print("2: Modeli Test Et")
    print("3: Yeni Yüz Tanı")

    choice = input("Seçim yap (1-3): ")

    if choice == "1":
        print("\nModel Eğitiliyor...\n")
        Train.train_model()
    elif choice == "2":
        print("\nModel Test Ediliyor...\n")
        Test.test_model()
    elif choice == "3":
        image_path = input("Tahmin edilecek yüzün dosya yolunu gir: ")
        print("Yüz Tanıma Sonucu:")
        Predict.predict_face(image_path)
    else:
        print("Geçersiz seçim!")

if __name__ == "__main__":
    main()
