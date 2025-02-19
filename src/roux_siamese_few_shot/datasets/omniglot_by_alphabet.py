""" A dataset for pytorch using the Omniglot dataset by Alphabet.

This is a variation on the Pytorch Omniglot Alphabet that sets class based on alphabet
instead of character.
"""
from os.path import join
from typing import Any, Callable, List, Optional, Tuple

from PIL import Image

from torchvision.datasets.utils import check_integrity, download_and_extract_archive, \
    list_dir, list_files
from torchvision.datasets.vision import VisionDataset


class OmniglotByAlphabet(VisionDataset):
    """A dataset for pytorch using the Omniglot dataset by Alphabet."""
    folder = "omniglot-py"
    download_url_prefix = "https://raw.githubusercontent.com/" \
                          "brendenlake/omniglot/master/python"
    zips_md5 = {
        "images_background": "68d2efa1b9178cc56df9314c21c6e718",
        "images_evaluation": "6b91aef0f799c5bb55b94e3f2daec811",
    }

    def __init__(
            self,
            root: str,
            background: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super().__init__(join(root, self.folder), transform=transform,
                         target_transform=target_transform)
        self.background = background

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. "
                "You can use download=True to download it"
            )

        self.target_folder = join(self.root, self._get_target_folder())
        self._alphabets = list_dir(self.target_folder)
        self._characters: List[str] = sum(
            ([(join(a, c), idx) for c in list_dir(join(self.target_folder, a))] for
             idx, a in enumerate(self._alphabets)), []
        )
        self._character_images = [
            [(image, char_label[1], idx) for image in
             list_files(join(self.target_folder, char_label[0]), ".png")]
            for idx, char_label in enumerate(self._characters)
        ]
        self._flat_character_images: List[Tuple[str, int, int]] = sum(
            self._character_images,
            []
        )

    def __len__(self) -> int:
        return len(self._flat_character_images)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image_name, alphabet_class, character_class = self._flat_character_images[index]
        image_path = join(self.target_folder, self._characters[character_class][0],
                          image_name)
        image = Image.open(image_path, mode="r").convert("L")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            alphabet_class = self.target_transform(alphabet_class)

        return image, alphabet_class

    def _check_integrity(self) -> bool:
        zip_filename = self._get_target_folder()
        if not check_integrity(join(self.root, zip_filename + ".zip"),
                               self.zips_md5[zip_filename]):
            return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        filename = self._get_target_folder()
        zip_filename = filename + ".zip"
        url = self.download_url_prefix + "/" + zip_filename
        download_and_extract_archive(url, self.root, filename=zip_filename,
                                     md5=self.zips_md5[filename])

    def _get_target_folder(self) -> str:
        return "images_background" if self.background else "images_evaluation"
