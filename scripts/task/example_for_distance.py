import distance

def main():
    distance_mm = distance.estimate_distance()
    print(f"测量的距离: {distance_mm / 10:.1f} cm")  # 转换为厘米输出

if __name__ == "__main__":
    main()