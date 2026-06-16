# #编译器与编译选项
# CXX = g++
# # -Wall 开启所有警告，-O3 开启最高级别优化（对跑神经网络非常重要！），-I 指定头文件目录
# CXXFLAGS = -Wall -O3 -I$(INC_DIR)

# #目录结构定义
# SRC_DIR = src
# INC_DIR = include
# BUILD_DIR = build

# #文件查找与替换
# #自动找到 src/ 下所有的 .cpp 文件
# SRCS = $(wildcard $(SRC_DIR)/*.cpp)
# # 将 src/xxx.cpp 替换为 build/xxx.o (编译的中间文件)
# OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCS))

# #最终输出的可执行文件路径
# TARGET = $(BUILD_DIR)/my_cnn

# # ==========================================
# # 编译规则
# # ==========================================

# # 默认目标 (输入 make 时默认执行这个)
# all: $(TARGET)

# # 链接规则：把所有的 .o 文件链接成最终的 my_cnn可执行文件
# $(TARGET): $(OBJS)
# 	@mkdir -p $(BUILD_DIR)
# 	$(CXX) $(CXXFLAGS) -o $@ $^
# 	@echo "=> 编译成功！可执行文件已生成: $(TARGET)"

# # 编译规则：把每个 .cpp 编译成单独的 .o 文件放在 build 目录下
# $(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
# 	@mkdir -p $(BUILD_DIR)
# 	$(CXX) $(CXXFLAGS) -c $< -o $@

# clean:
# 	rm -rf $(BUILD_DIR)
# 	@echo "=> 清理完成！已删除 $(BUILD_DIR) 目录。"

# # 声明伪目标（防止当前目录下正好有叫 all 或 clean 的文件引起冲突）
# .PHONY: all clean