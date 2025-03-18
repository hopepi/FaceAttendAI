import tkinter as tk
from tkinter import ttk, filedialog

root = tk.Tk()
root.title("Yüz Tespiti ile Yoklama Sistemi")
root.geometry("450x500")

selected_source = tk.StringVar()
selected_source.set("Canlı Kamera")

def open_database():
    folder_path = filedialog.askdirectory(title="Veri Klasörü Seçiniz")
    if folder_path:
        database_label.config(text=f"Seçilen Klasör:\n{folder_path}")
        database_label.selected_path = folder_path
        live_button.config(state="normal")
        individual_button.config(state="normal")

def select_excel_file(label):
    file_path = filedialog.askopenfilename(
        title="Kişiler Excel Dosyasını Seç",
        filetypes=[("Excel Dosyası", "*.xlsx"), ("CSV Dosyası", "*.csv"), ("Tüm Dosyalar", "*.*")]
    )
    if file_path:
        label.config(text=f"Seçilen Kişiler Dosyası:\n{file_path}")

def open_live_summary():
    root.withdraw()
    new_window = tk.Toplevel()
    new_window.title("Canlı Tekte Özet Çıkarma")
    new_window.geometry("400x500")

    tk.Label(new_window, text="Canlı Tekte Özet Çıkarma Modu Açıldı!", font=("Helvetica", 12)).pack(pady=10)

    tk.Label(new_window, text="Kaynak Seçiniz:", font=("Helvetica", 10)).pack()
    source_options = ["Canlı Kamera", "Video Dosyası", "Resim Dosyası"]
    source_selection = ttk.Combobox(new_window, textvariable=selected_source, values=source_options, state="readonly")
    source_selection.pack(pady=5)

    tk.Label(new_window, text="Ders İsmi:", font=("Helvetica", 10)).pack()
    course_name_entry = tk.Entry(new_window, width=30)
    course_name_entry.pack(pady=5)

    tk.Label(new_window, text="Kişiler Excel Dosyanızı Seçin:", font=("Helvetica", 10)).pack()
    excel_label = tk.Label(new_window, text="Henüz seçilmedi", font=("Helvetica", 10), fg="red")
    excel_label.pack(pady=5)

    tk.Button(new_window, text="Kişiler Excel Dosyası Seç", command=lambda: select_excel_file(excel_label)).pack(pady=5)

    new_window.protocol("WM_DELETE_WINDOW", lambda: reopen_main(new_window))

def open_individual_recognition():
    root.withdraw()
    new_window = tk.Toplevel()
    new_window.title("Tek Tek Tanımlama")
    new_window.geometry("400x500")

    tk.Label(new_window, text="Tek Tek Tanımlama Modu Açıldı!", font=("Helvetica", 12)).pack(pady=10)

    tk.Label(new_window, text="Kaynak Seçiniz:", font=("Helvetica", 10)).pack()
    source_options = ["Canlı Kamera", "Video Dosyası", "Resim Dosyası"]
    source_selection = ttk.Combobox(new_window, textvariable=selected_source, values=source_options, state="readonly")
    source_selection.pack(pady=5)

    tk.Label(new_window, text="Ders İsmi:", font=("Helvetica", 10)).pack()
    course_name_entry = tk.Entry(new_window, width=30)
    course_name_entry.pack(pady=5)

    tk.Label(new_window, text="Kişiler Excel Dosyanızı Seçin:", font=("Helvetica", 10)).pack()
    excel_label = tk.Label(new_window, text="Henüz seçilmedi", font=("Helvetica", 10), fg="red")
    excel_label.pack(pady=5)

    tk.Button(new_window, text="Kişiler Excel Dosyası Seç", command=lambda: select_excel_file(excel_label)).pack(pady=5)

    new_window.protocol("WM_DELETE_WINDOW", lambda: reopen_main(new_window))

def reopen_main(window):
    window.destroy()
    root.deiconify()

label = tk.Label(root, text="Hoşgeldiniz", font=("Helvetica", 14, "bold"))
label.pack(pady=10)

database_button = tk.Button(root, text="Veri Klasörü Seç", width=20, command=open_database)
database_button.pack(pady=10)

database_label = tk.Label(root, text="Henüz klasör seçilmedi.", font=("Helvetica", 10), fg="red")
database_label.pack(pady=5)
database_label.selected_path = ""

model_list = ["FaceNet", "FaceNet2", "FaceNet3"]
selected_model = tk.StringVar()
selected_model.set(model_list[0])

ttk.Label(root, text="Model Seçiniz:", font=("Helvetica", 12)).pack(pady=5)
model_selection = ttk.Combobox(root, textvariable=selected_model, values=model_list, state="readonly")
model_selection.pack(pady=5)

live_button = tk.Button(root, text="Canlı Özet Çıkarma", width=20, command=open_live_summary, state="disabled")
live_button.pack(pady=10)

individual_button = tk.Button(root, text="Tek Tek Tanımlama", width=20, command=open_individual_recognition, state="disabled")
individual_button.pack(pady=10)

github_label_1 = tk.Label(root, text="https://github.com/hopepi", fg="blue", cursor="hand2")
github_label_1.pack(pady=5)

github_label_2 = tk.Label(root, text="https://github.com/EfeCyber", fg="blue", cursor="hand2")
github_label_2.pack(pady=5)

root.mainloop()
