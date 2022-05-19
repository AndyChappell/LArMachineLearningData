function download() {
   wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=$1' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$1" -O $2 && rm -rf /tmp/cookies.txt
}

# Google Drive links have the form https://drive.google.com/file/d/FILEID/view?usp=sharing
# You want the FILEID
# Note: For some reason the above function doesn't work for the files, so you need to set the two &id=FILEID instances in the link

if [ -z $MY_TEST_AREA ]
then
  echo "MY_TEST_AREA is not set, can't download the files"
  return 1
fi

if [ ! -d $MY_TEST_AREA/LArMachineLearningData/ ]
then
  echo "LArMachineLearningData does not exist in MY_TEST_AREA: $MY_TEST_AREA, Not downloading the files"
  return 1
fi

dune=0
uboone=0
sbnd=0

if [ "$1" == "all" ]; then
  dune=1
  uboone=1
  sbnd=1
elif [ "$1" == "dune" ]; then
  dune=1
elif [ "$1" == "uboone" ]; then
  uboone=1
elif [ "$1" == "sbnd" ]; then
  sbnd=1
else
  echo "Please specify the experiment [ all | dune | uboone | sbnd"
  return 1
fi

### PandoraMVAData
mkdir -p $MY_TEST_AREA/LArMachineLearningData/PandoraMVAData
cd $MY_TEST_AREA/LArMachineLearningData/PandoraMVAData

if [ "$uboone" == "1" ]; then
  # MicroBooNE
  wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1b3m9Glj1Qjx5tnSdvqLIBNFBWpH8NrvX' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1b3m9Glj1Qjx5tnSdvqLIBNFBWpH8NrvX" -O PandoraSvm_v03_11_00.xml && rm -rf /tmp/cookies.txt
fi

if [ "$dune" == "1" ]; then
  # DUNE
  wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1i5mi545-NQU15raIyYOwbWY8VQuCv8Ox' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1i5mi545-NQU15raIyYOwbWY8VQuCv8Ox" -O PandoraBdt_BeamParticleId_ProtoDUNESP_v03_26_00.xml && rm -rf /tmp/cookies.txt
  wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1eeiScUaLjEoACxCyb73JwGnuMsJ1Y4Iq' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1eeiScUaLjEoACxCyb73JwGnuMsJ1Y4Iq" -O PandoraBdt_PfoCharacterisation_ProtoDUNESP_v03_26_00.xml && rm -rf /tmp/cookies.txt
  wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=12cl3gpijkcpIIZZGeThtH2e4Vzvf6Xqu' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=12cl3gpijkcpIIZZGeThtH2e4Vzvf6Xqu" -O PandoraBdt_Vertexing_ProtoDUNESP_v03_26_00.xml && rm -rf /tmp/cookies.txt

  wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1oMDHwtOapNcs8m4H29MpQkiKbFo0rWbT' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1oMDHwtOapNcs8m4H29MpQkiKbFo0rWbT" -O PandoraBdt_PfoCharacterisation_DUNEFD_v03_26_00.xml && rm -rf /tmp/cookies.txt
  wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=17XqHdu53btjfnRF4-mHgv5mpE1HoDpDT' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=17XqHdu53btjfnRF4-mHgv5mpE1HoDpDT" -O PandoraBdt_Vertexing_DUNEFD_v03_27_00.xml && rm -rf /tmp/cookies.txt
fi

### PandoraMVAs
mkdir -p $MY_TEST_AREA/LArMachineLearningData/PandoraMVAs
cd $MY_TEST_AREA/LArMachineLearningData/PandoraMVAs

if [ "$sbnd" == "1" ]; then
# SBND
  wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1lGn-_BCK9TpEdVZUElAAxFJ9ynazcCY7' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1lGn-_BCK9TpEdVZUElAAxFJ9ynazcCY7" -O PandoraBdt_v09_32_00_SBND.xml && rm -rf /tmp/cookies.txt
fi



### PandoraNetworkData
mkdir -p $MY_TEST_AREA/LArMachineLearningData/PandoraNetworkData
cd $MY_TEST_AREA/LArMachineLearningData/PandoraNetworkData

if [ "$dune" == "1" ]; then
  # DUNE
  download "1EOmyofKFl9NAZqRh-z34bY7-M9lXvG9U" "PandoraUnet_TSID_DUNEFD_U_v03_25_00.pt"
  download "1NqeKHnCkWdTIKw6CLqUZp4l-q7EIkAQa" "PandoraUnet_TSID_DUNEFD_V_v03_25_00.pt"
  download "1kWfT8d4GtlMsoxh6cH0quIUNRdHSClhj" "PandoraUnet_TSID_DUNEFD_W_v03_25_00.pt"
fi

