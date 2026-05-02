# 🛸 NEBULA: Hexacopter ROS2 PX4 Simulation
![License](https://img.shields.io/badge/License-MIT-blue.svg)
![ROS2](https://img.shields.io/badge/ROS2-Humble/Foxy-orange)
![PX4](https://img.shields.io/badge/PX4-Autopilot-green)

This repository contains the full simulation stack for an autonomous Hexacopter platform, developed by **Team NEBULA**. The project integrates **ROS2** and **PX4 Autopilot** to achieve high-precision flight control and real-time AI-driven threat detection.

<p align="center">
  <img src="images/hexacopter.png" width="700" alt="Hexacopter Banner">
</p>

---

## 🚀 Key Features
*   **SITL Simulation:** Advanced Software-In-The-Loop simulation using PX4 and Gazebo.
*   **ROS2 Integration:** Full control and telemetry via ROS2 nodes.
*   **AI Vision:** Real-time object detection and classification using **YOLOv8s**.
*   **Threat Assessment:** Automated module for identifying and prioritizing tactical threats.

---

## 🛠 System Architecture
The system is built on a modular architecture to ensure scalability and reliability in autonomous missions.

| Component | Description |
| :--- | :--- |
| **UAV Frame** | Hexacopter (High stability & Payload capacity) |
| **Flight Stack** | PX4 Autopilot with MAVROS/Micro-XRCE-DDS |
| **Mission Logic** | Custom ROS2 Python/C++ Nodes |
| **Perception** | YOLOv8s optimized for aerial viewpoints |
| **Environment** | Gazebo / Ignition Simulation |

---

## 📽 Simulation Demo
Check out the hexacopter's flight performance and system stability during simulation runs:

![Hexacopter Simulation](images/demo.gif)

---

## 🔍 Threat Detection Module
Our perception pipeline is designed to identify potential risks in real-time. Below is a snapshot of the **YOLOv8s** inference running during a tactical flight mission.

<p align="center">
  <img src="images/ThreatDetection.png" width="600" alt="Threat Detection Analysis">
</p>

---

## 🔗 Project Showcase
For a comprehensive technical walkthrough, including flight tests and implementation details, visit our professional showcase:

👉 **[Watch the Full Project Video on LinkedIn](https://www.linkedin.com/posts/abdelfattah-ahmed7_px4-yolov8s-robotics-ugcPost-7450828759437369344-IxIU?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFNVdBgBLRXykO2ahcmCnoWAi96H6J2EXbs)**

---

## 👥 The Team: NEBULA
This project was developed by the **NEBULA** team, focusing on pushing the boundaries of Autonomous UAV technology.

*   **Lead Software & Simulation:** [Abdelfattah Ahmed Abdelfattah](https://github.com/abdelfatah7)
*   **Faculty:** Engineering, New Mansoura University.

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
