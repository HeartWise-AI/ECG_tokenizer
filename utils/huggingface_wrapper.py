import os
from huggingface_hub import snapshot_download, HfApi

class HuggingFaceWrapper:
    hugging_face_api_key = None
    api = HfApi()

    @staticmethod
    def get_model(
        repo_id: str, 
        local_dir: str, 
        hugging_face_api_key: str
    )->str:           
        # Download repo from HuggingFace
        os.makedirs(local_dir, exist_ok=True)
        print(f"Downloading {repo_id} to {local_dir}")
        local_dir = snapshot_download(repo_id=repo_id, local_dir=local_dir, repo_type="model", token=hugging_face_api_key)
        
        print(f"{repo_id} downloaded to {local_dir}")

        return local_dir

    @classmethod
    def upload_model(
        cls,
        repo_id: str, 
        local_dir: str, 
    )->None:
        """
        Upload a model to Hugging Face.
        
        :param repo_id: The ID of the repository (e.g., 'username/repo-name')
        :param local_dir: The local directory containing the model files
        :param commit_message: The commit message for this update
        """
        try:
            # Ensure the repository exists (create if it doesn't)
            try:
                cls.api.create_repo(repo_id=repo_id, token=cls.hugging_face_api_key, exist_ok=True)
            except Exception as e:
                print(f"Note: {e}")

            # Upload all files in the directory
            for root, _, files in os.walk(local_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    repo_path = os.path.relpath(file_path, local_dir)
                    
                    cls.api.upload_file(
                        path_or_fileobj=file_path,
                        path_in_repo=repo_path,
                        repo_id=repo_id,
                        token=cls.hugging_face_api_key,
                        commit_message=f"Upload {file}"
                    )
                    print(f"Uploaded {file} to {repo_id}")

            print(f"Successfully uploaded model to {repo_id}")
        except Exception as e:
            print(f"An error occurred while uploading the model: {e}")