for r in 32 64 128 192 256; do
  python dtypeProbe.py --kh 99.7 --N 33 --p 10 --leaf 0.03125 \
      --leaf_size 800 --rank $r 2>&1 | grep "RIGHT ERR" | head -1
done
