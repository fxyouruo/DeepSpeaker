import logging
import subprocess

from setuptools import setup, find_packages
from setuptools.command.install import install


class CustomCommands(install):
    @staticmethod
    def run_custom_command(command_list):
        p = subprocess.Popen(
            command_list,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        stdout_data, _ = p.communicate()
        logging.info('Log command output: %s', stdout_data)
        if p.returncode != 0:
            raise RuntimeError('Command %s failed: exit code: %s' % (command_list, p.returncode))

    def run(self):
        try:
            # workaround for third party libraries on gcloud
            self.run_custom_command(['apt-get', 'update'])
            self.run_custom_command(['apt-get', 'install', '-y', 'libhdf5-dev'])
            self.run_custom_command(['apt-get', 'install', '-y', 'build-essential'])
            self.run_custom_command(['apt-get', 'install', '-y', 'libav-tools'])
            self.run_custom_command(['apt-get', 'install', '-y', 'mediainfo'])
        except:
            pass
        install.run(self)


if __name__ == '__main__':
    REQUIRED_PACKAGES = ['numpy', 'keras', 'tensorflow', 'librosa', 'pydub', 'h5py', 'python_speech_features']
    setup(
        name='DeepSpeaker',
        version='0.0.1',
        description='DeepSpeaker Packages',
        long_description="DeepSpeaker Packages",
        author='SeerNet',
        author_email='engineering@seernet.io',
        url='https://github.com/SEERNET/DeepSpeaker',
        install_requires=REQUIRED_PACKAGES,
        packages=find_packages(),
        include_package_data=True,
        cmdclass={
            'install': CustomCommands,
        })

