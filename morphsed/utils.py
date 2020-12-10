import pathlib

__all__ = ['package_path']

package_path = pathlib.Path(__file__).parent.absolute()
