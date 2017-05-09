for /l %%x in (0.1, 0.1, 1) do (
echo python gridworld.py -a q -k 50 -n 0 -g BridgeGrid -e %%x > %%x.txt

)