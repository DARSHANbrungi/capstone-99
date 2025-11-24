import os

# The root directory for the project
project_name = "iot_anomaly_detection"

# List of all directories to be created
list_of_dirs = [
    os.path.join(project_name, "data", "raw"),
    os.path.join(project_name, "data", "processed"),
    os.path.join(project_name, "notebooks"),
    os.path.join(project_name, "src", "components"),
    os.path.join(project_name, "src", "pipeline"),
    os.path.join(project_name, "src", "utils"),
    os.path.join(project_name, "saved_models"),
    os.path.join(project_name, "templates"),
    "logs" # Logs directory at the root level
]

# Create all the directories
for dir_path in list_of_dirs:
    os.makedirs(dir_path, exist_ok=True)
    print(f"Created directory: {dir_path}")

# List of all files to be created
list_of_files = [
    os.path.join(project_name, "data", "raw", ".gitkeep"), # Use .gitkeep to track empty dirs
    os.path.join(project_name, "data", "processed", ".gitkeep"),
    os.path.join(project_name, "notebooks", "1-data-exploration.ipynb"),
    os.path.join(project_name, "src", "components", "__init__.py"),
    os.path.join(project_name, "src", "components", "data_ingestion.py"),
    os.path.join(project_name, "src", "components", "data_transformation.py"),
    os.path.join(project_name, "src", "components", "model_trainer.py"),
    os.path.join(project_name, "src", "components", "model_evaluator.py"),
    os.path.join(project_name, "src", "pipeline", "__init__.py"),
    os.path.join(project_name, "src", "pipeline", "training_pipeline.py"),
    os.path.join(project_name, "src", "pipeline", "prediction_pipeline.py"),
    os.path.join(project_name, "src", "utils", "__init__.py"),
    os.path.join(project_name, "src", "utils", "utils.py"),
    os.path.join(project_name, "src", "__init__.py"),
    os.path.join(project_name, "src", "exception.py"),
    os.path.join(project_name, "src", "logger.py"),
    os.path.join(project_name, "saved_models", ".gitkeep"),
    os.path.join(project_name, "templates", "index.html"),
    os.path.join(project_name, "app.py"),
    os.path.join(project_name, "requirements.txt"),
    os.path.join(project_name, "setup.py")
]

# Create all the empty files
for filepath in list_of_files:
    # Use 'w' to create an empty file
    with open(filepath, "w") as f:
        pass # Just create the file, no content needed for now
    print(f"Created empty file: {filepath}")

print("\nProject structure created successfully!")
print(f"Next step: Move your 'combined_dataset.csv' into '{os.path.join(project_name, 'data', 'raw')}'")

