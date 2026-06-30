# Detect Pixelated Image and Correct It

An AI-powered deep learning project that automatically detects whether an image is pixelated and restores its visual quality using Convolutional Neural Networks (CNN) and an Enhanced U-Net architecture.

---

## 📖 Overview

Pixelated images often lose important visual details when enlarged, making them unsuitable for modern applications. This project uses deep learning to first detect pixelation and then restore image quality by reconstructing missing details.

The system improves image clarity and produces visually enhanced outputs that can be used in various digital applications.

---

## 🚀 Features

- Detects whether an input image is pixelated
- Classifies images into pixelated or non-pixelated
- Restores pixelated images using deep learning
- Improves image sharpness and visual quality
- End-to-end automated image enhancement pipeline

---

## 🧠 Problem Statement

Low-resolution images become blurry and lose important details when enlarged. Traditional image processing techniques struggle to recover missing information effectively.

This project addresses this problem by using deep learning models to detect pixelation and reconstruct high-quality images.

---

## 💡 Solution

The proposed solution consists of two stages:

1. **Pixelation Detection**
   - A CNN model analyzes the input image.
   - Determines whether the image is pixelated.

2. **Image Restoration**
   - If the image is pixelated, it is passed to an Enhanced U-Net model.
   - The model reconstructs missing details and generates a clearer, high-resolution output.

---

## 🔄 Workflow

```text
Input Image
      │
      ▼
CNN Detection Model
      │
      ▼
Is Image Pixelated?
      │
 ┌────┴────┐
 │         │
No        Yes
 │         │
 ▼         ▼
Output   Enhanced U-Net
             │
             ▼
      Restored Image
```

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- Scikit-learn
- Google Cloud Platform

---

## 📂 Project Structure

```
Pixelated-Image-Detection/
│
├── dataset/
├── models/
├── notebooks/
├── src/
├── outputs/
├── images/
├── requirements.txt
├── README.md
└── app.py
```

---

## 🎯 Applications

- Image Super Resolution
- Digital Photography
- Social Media Image Enhancement
- Medical Imaging
- Satellite Image Processing
- Surveillance Systems
- Image Restoration
- Graphic Design

---

## 📈 Future Improvements

- Support real-time video enhancement
- Improve restoration quality using GAN-based models
- Deploy as a web application
- Optimize for mobile devices
- Train on larger and more diverse datasets
- Reduce inference time for faster predictions

---

## 👨‍💻 Team Members

- Praveen Raj N
- Lingeshwar G.J
- Nishanth
- Mrithul Snehal
- John Osborne

---

## 🙋 My Contribution

- Researched deep learning algorithms for pixelation detection.
- Implemented and optimized CNN-based detection models.
- Developed image restoration pipeline.
- Tested and evaluated model performance.
- Collaborated on integrating the complete system.

---

## 📌 Results

The proposed deep learning pipeline successfully:

- Detects pixelated images accurately.
- Restores image details effectively.
- Enhances overall visual quality.
- Produces cleaner and sharper images suitable for practical applications.

---

## 🔮 Future Scope

This project can be extended into:

- AI-based photo restoration software
- Video enhancement systems
- Cloud-based image enhancement APIs
- Mobile image enhancement applications
- AI-powered editing tools

---

## 🤝 Contributing

Contributions are welcome!

Feel free to fork the repository, create a new branch, and submit a pull request.

---

## 📜 License

This project is developed for educational and research purposes.

---

⭐ If you found this project useful, consider giving it a star!
