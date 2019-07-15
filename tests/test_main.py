import os
import subprocess

import numpy as np
import pytest

from PIL import Image
from yaml import load
from ecodse_funtime_alpha.main import get_args


class TestArgparse(object):

    def test_argparsenormal(self, tmpdir):
        hp_yaml = tmpdir.join("hyperparam.yml")
        fakelr = 0.01
        fakefilter1 = 64
        fakedataaugment = "cifar10"
        hp_yaml.write_text(f"dataaugment: {fakedataaugment}\nlr: {fakelr}\nfilter1: {fakefilter1}", encoding="utf-8")

        fakearg = ['--imagepath=./', '--labelpath=fakedir/name.csv',
                   '--seed=1', f'--config={str(tmpdir.join("hyperparam.yml"))}'
                   ]
        args = get_args(fakearg)
        assert args.imagepath == './'
        assert args.labelpath == 'fakedir/name.csv'
        assert args.seed == 1
        with open(args.config, 'r') as stream:
            hp = load(stream)

        assert hp['filter1'] == fakefilter1
        assert hp['dataaugment'] == fakedataaugment
        assert hp['lr'] == fakelr

    @pytest.mark.xfail(raises=SystemExit)
    def test_argparse_lr(self):
        fakearg = ['--lr=a', '--config=dummy']
        _ = get_args(fakearg)

    @pytest.mark.xfail(raises=SystemExit)
    def test_argparse_seed(self):
        fakearg = ['--seed=a', '--config=dummy']
        _ = get_args(fakearg)

    def test_argparse_imagepath(self):
        fakearg = ['--imagepath=notavalidpath', '--config=dummy']
        args = get_args(fakearg)
        assert not os.path.isdir(args.imagepath)

    def test_argparse_labelpath(self):
        fakearg = ['--labelpath=invalid.csv', '--config=dummy']
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

        nepoch = 3
        batchsize = 1
        patience = 1
        lr = 1
        hp_yaml = tmpdir.join("hyperparam.yml")
        yaml_text = f"dataaugment: cifar10\nlr: {lr}\nnepoch: {nepoch}\npatience: {patience}\nbatchsize: {batchsize}\nfilter1: 64\nksize1: 3\nstride1: 1\n"
        yaml_text += "pool1: 2\nstridepool1: 2\nfilter2: 128\nksize2: 3\nstride2: 1\npool2: 2\nstridepool2: 2\nfilter3: 256\nksize3: 3\n"
        yaml_text += "stride3: 1\npool3: 2\nstridepool3: 2\nfilter4: 512\nksize4: 3\nstride4: 1\npool4: 2\nstridepool4: 2\n"
        yaml_text += "filter5: 512\nksize5: 3\nstride5: 1\npool5: 2\nstridepool5: 2\ndense1: 4096\ndense2: 4096"
        hp_yaml.write_text(yaml_text, encoding="utf-8")

    def test_main(self, tmpdir):
        imagedir = str(tmpdir.join("train-jpg"))
        labelpath = str(tmpdir.join("train-jpg").join("train_v2.csv"))
        ymlpath = str(tmpdir.join("hyperparam.yml"))
        fakearg = ["--imagepath", imagedir, "--labelpath", labelpath, "--config", ymlpath, "--log", "test.log"]

        assert subprocess.call(["python", "ecodse_funtime_alpha/main.py", *fakearg]) == 0
