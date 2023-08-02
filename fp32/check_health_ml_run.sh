echo "#############################################"
echo "Current time :"
date
{
	lsof -i tcp:65505 &&
	echo API is FINE
} || {
	echo RESTART
	nohup python3 /home/bentoml/bento/src/model_run.py > /home/bentoml/bento/src/service.log &
}
