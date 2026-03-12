import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import onnxruntime as ort
import numpy as np
import torchvision.transforms as transforms
import os

MODEL_PATH = "model/iris-vit.onnx"
CLASSES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

# Define preprocessing (matches the original Colab code)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class RetinaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Diabetic Retinopathy Detection")
        self.root.geometry("600x700")
        self.root.configure(bg="#f0f0f0")

        # Load the ONNX session
        self.session = self.load_model()

        # UI Elements
        self.title_label = tk.Label(root, text="Diabetic Retinopathy Analyzer", font=("Helvetica", 18, "bold"), bg="#f0f0f0")
        self.title_label.pack(pady=20)

        # Image display area
        self.canvas = tk.Canvas(root, width=400, height=400, bg="white", relief=tk.SUNKEN, bd=2)
        self.canvas.pack(pady=10)
        self.image_on_canvas = None

        # Upload button
        self.upload_btn = tk.Button(root, text="Upload Fundus Image", command=self.upload_and_predict, font=("Helvetica", 12), bg="#4CAF50", fg="white", padx=10, pady=5)
        self.upload_btn.pack(pady=10)

        # Result labels
        self.result_label = tk.Label(root, text="Prediction: ---", font=("Helvetica", 16, "bold"), bg="#f0f0f0", fg="blue")
        self.result_label.pack(pady=5)

        self.confidence_label = tk.Label(root, text="Confidence: ---", font=("Helvetica", 12), bg="#f0f0f0")
        self.confidence_label.pack(pady=5)

        self.details_label = tk.Label(root, text="", font=("Courier", 10), bg="#f0f0f0", justify=tk.LEFT)
        self.details_label.pack(pady=10)

    def load_model(self):
        """Loads the ONNX model from the local 'model' directory."""
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror("Model Missing", f"Could not find model at:\n{MODEL_PATH}\n\nPlease create a 'model' folder and place 'iris-vit.onnx' inside it.")
            return None
        try:
            return ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        except Exception as e:
            messagebox.showerror("Error Loading Model", str(e))
            return None

    def upload_and_predict(self):
        """Handles file selection, preprocessing, and prediction."""
        if self.session is None:
            messagebox.showerror("Error", "Model is not loaded. Please fix the model path and restart.")
            return

        # 1. Open File Dialog
        file_path = filedialog.askopenfilename(
            title="Select a Fundus Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if not file_path:
            return

        try:
            # 2. Load and display image
            img = Image.open(file_path).convert("RGB")
            self.display_image(img)

            # 3. Preprocess image
            input_tensor = transform(img).unsqueeze(0).numpy().astype(np.float32)

            # 4. Run Inference
            outputs = self.session.run(None, {"input": input_tensor})[0][0]

            # 5. Process Softmax & Results
            exp_scores = np.exp(outputs)
            probs = exp_scores / np.sum(exp_scores)
            pred_idx = np.argmax(probs)

            # 6. Update UI
            self.result_label.config(text=f"Prediction: {CLASSES[pred_idx]}")
            self.confidence_label.config(text=f"Confidence: {probs[pred_idx]*100:.1f}%")

            # Format detailed probabilities
            details = "Full Probabilities:\n" + "-"*30 + "\n"
            for name, p in zip(CLASSES, probs):
                details += f"{name:<20} {p*100:5.1f}%\n"
            self.details_label.config(text=details)

        except Exception as e:
            messagebox.showerror("Error Processing Image", str(e))

    def display_image(self, img):
        """Resizes and displays the image on the Tkinter canvas."""
        # Resize for display purposes while keeping aspect ratio
        img.thumbnail((400, 400), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(img)
        
        # Center image on canvas
        self.canvas.delete("all")
        x_offset = (400 - img.width) // 2
        y_offset = (400 - img.height) // 2
        self.image_on_canvas = self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self.tk_image)

if __name__ == "__main__":
    root = tk.Tk()
    app = RetinaApp(root)
    root.mainloop()

