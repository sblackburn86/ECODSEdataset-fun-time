import os
import subprocess

import numpy as np
import pytest

from PIL import Image
from ecodse_funtime_alpha.main import get_args


class TestArgparse(object):

    def test_argparsenormal(self):
        fakearg = ['--imagepath=./', '--labelpath=fakedir/name.csv',
                   '--seed=1', '--kernels=10', '--ksize=1',
                   '--lr=0.01', '--nepoch=2', '--batchsize=4'
                   ]
        args = get_args(fakearg)
        assert args.imagepath == './'
        assert args.labelpath == 'fakedir/name.csv'
        assert args.seed == 1
        assert args.kernels == 10
        assert args.ksize == 1
        assert args.lr == 0.01
        assert args.nepoch == 2
        assert args.batchsize == 4

    @pytest.mark.xfail(raises=SystemExit)
    def test_argparse_lr(self):
        fakearg = ['--lr=a']
        _ = get_args(fakearg)

    @pytest.mark.xfail(raises=SystemExit)
    def test_argparse_seed(self):
        fakearg = ['--seed=a']
        _ = get_args(fakearg)

    def test_argparse_imagepath(self):
        fakearg = ['--imagepath=notavalidpath']
        args = get_args(fakearg)
        assert not os.path.isdir(args.imagepath)

    def test_argparse_labelpath(self):
        fakearg = ['--labelpath=invalid.csv']
        args = get_args(fakearg)
        assert not os.path.exists(args.labelpath)


class TestMain(object):
    @pytest.fixture(autouse=True)
    def mock_files(self, tmpdir):
        self.img_size = 256
        self.img_channel = 3
        self.img_colorpixel = 128
        self.img_color = 255

        p = tmpdir.mkdir("train-jpg").join("train_v2.csv")
        csv_text = ["image_name,tags\n"] + [f"train_{x}, a b c d e f g h\n" for x in range(9)] + ["train_9, a b c"]
        p.write_text("".join(csv_text), encoding="utf-8")

        img = np.zeros((self.img_size, self.img_size, self.img_channel), dtype=np.uint8)
        img[self.img_colorpixel] = self.img_color

        for x in range(10):
            Image.fromarray(img).save(str(tmpdir.join("train-jpg").join(f"train_{x}.jpg")), quality=100)

    def test_main(self, tmpdir):
        imagedir = str(tmpdir.join("train-jpg"))
        labelpath = str(tmpdir.join("train-jpg").join("train_v2.csv"))
        nepoch = 1
        batchsize = 1
        fakearg = ["--imagepath", imagedir, "--labelpath", labelpath, "--nepoch", str(nepoch)]
        fakearg = fakearg + ["--batchsize", str(batchsize), "--log", "test.log"]

        assert subprocess.call(["python", "ecodse_funtime_alpha/main.py", *fakearg]) == 0
