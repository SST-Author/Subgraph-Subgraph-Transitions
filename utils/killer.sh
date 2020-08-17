if [ -z "$1" ]; 
then
    echo "Must pass an argument to grep for in ps aux."
    exit
fi

ps aux | grep $1 | grep $USER | grep -v grep | awk '{print $2}' > /tmp/proc_to_kill.txt

cat /tmp/proc_to_kill.txt
echo "Killing the above pids..."

while read p; do
    kill $p
done < /tmp/proc_to_kill.txt
