FROM pytorch-lab

RUN conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

RUN conda install -c anaconda imageio matplotlib pandas

RUN conda install -c derickl torchtext

RUN python -m spacy download en

RUN conda install scikit-learn