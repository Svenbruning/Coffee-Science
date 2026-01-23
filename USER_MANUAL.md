# Setup Guide

# Installation Steps
## Docker Installation

### Windows
1. Download Docker Desktop for Windows from the official Docker website
2. Run the installer and follow the prompts
3. Restart computer if prompted
4. Open Docker Desktop and wait until it shows Docker is running

### macOS
1. Download Docker Desktop for Mac from the official Docker website (choose Apple 
Silicon or Intel based on your Mac).
2. Open the downloaded .dmg file and drag Docker to Applications
3. Launch Docker from Applications
4. Grant permissions when prompted
5. Wait until Docker shows Docker is running


## Get the project code 
Ensure the project code is available on your machine:
1. Download zip file
2. Extract all...


## Start Desktop Docker


## Go to the right directory
Open "Coffee-Science-initial-setup" that is inside the folder


## Open terminal window in Windows Powershell
Right click on "wildlands-forecast" -> Open in Terminal with the correct path


### Build the Docker Images
docker compose build


### Start the application
docker compose up -d


## Visit link http://localhost:8080


# Common Issues
• Docker not running: Make sure Docker Desktop is started before running 
commands.

• Port already in use: Stop other services using the same ports or update docker compose.yml.


# Stopping the Application
To stop all running containers:
docker compose down


# Summary
From the project folder
docker compose build
docker compose up -d
