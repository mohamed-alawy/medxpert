# Create release directory
$releaseDir = "medxpert-release"
Write-Host "Creating release directory: $releaseDir"
New-Item -ItemType Directory -Force -Path $releaseDir

# Create necessary subdirectories
Write-Host "Creating subdirectories..."
New-Item -ItemType Directory -Force -Path "$releaseDir/templates"
New-Item -ItemType Directory -Force -Path "$releaseDir/static"
New-Item -ItemType Directory -Force -Path "$releaseDir/static/css"
New-Item -ItemType Directory -Force -Path "$releaseDir/static/js"
New-Item -ItemType Directory -Force -Path "$releaseDir/static/uploads"
New-Item -ItemType Directory -Force -Path "$releaseDir/models"

# Copy main files
Write-Host "Copying main files..."
Copy-Item "app.py" -Destination "$releaseDir/"
Copy-Item "requirements.txt" -Destination "$releaseDir/"
Copy-Item "README.md" -Destination "$releaseDir/"
Copy-Item ".gitignore" -Destination "$releaseDir/"

# Copy templates
Write-Host "Copying template files..."
Copy-Item "templates/*.html" -Destination "$releaseDir/templates/"

# Copy static files
Write-Host "Copying static files..."
Copy-Item "static/css/*" -Destination "$releaseDir/static/css/" -ErrorAction SilentlyContinue
Copy-Item "static/js/*" -Destination "$releaseDir/static/js/" -ErrorAction SilentlyContinue
Copy-Item "static/uploads/.gitkeep" -Destination "$releaseDir/static/uploads/"

# Copy models README
Write-Host "Copying models README..."
Copy-Item "models/README.md" -Destination "$releaseDir/models/"

# Create .gitkeep files
Write-Host "Creating .gitkeep files..."
Set-Content -Path "$releaseDir/static/uploads/.gitkeep" -Value ""
Set-Content -Path "$releaseDir/models/.gitkeep" -Value ""

# Create ZIP file
Write-Host "Creating ZIP file..."
$date = Get-Date -Format "yyyyMMdd"
$zipFile = "medxpert-release-$date.zip"
Compress-Archive -Path "$releaseDir/*" -DestinationPath $zipFile -Force

Write-Host "`nRelease preparation completed!"
Write-Host "Release files are in: $releaseDir"
Write-Host "ZIP file created: $zipFile"
Write-Host "`nPlease verify the contents before distribution." 