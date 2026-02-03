import tree_sitter
import tree_sitter_java
import json
import os

class JavaCodeAnalyzer:
    def __init__(self):
        """
        初始化 Tree-sitter 解析器
        """
        try:
            # 初始化 Parser
            self.parser = tree_sitter.Parser()
            
            # 加载 Java 语言定义
            # 注意：不同版本的 tree-sitter 库加载方式略有不同，这里使用最新标准
            java_lang = tree_sitter.Language(tree_sitter_java.language())
            self.parser.language = java_lang
            
        except Exception as e:
            print(f"Error: 初始化 Tree-sitter 失败。请确保已安装 tree-sitter 和 tree-sitter-java。\n{e}")
            raise

    def analyze_code(self, source_code):
        """
        分析 Java 代码字符串
        """
        if not source_code:
            return {"classes": []}

        # 【核心修复】：将源码转换为 UTF-8 字节
        # Tree-sitter 的 start_byte/end_byte 是基于字节的，直接切片 Unicode 字符串会出错（特别是包含中文时）
        self.source_bytes = bytes(source_code, "utf8")
        
        # 解析代码生成语法树
        tree = self.parser.parse(self.source_bytes)
        
        result = {
            "classes": []
        }

        # 遍历根节点下的所有 class_declaration
        # 使用 cursor 遍历或者简单的递归查找
        root_node = tree.root_node
        self._find_and_process_classes(root_node, result["classes"])
        
        return result

    def _get_text(self, node):
        """
        【工具函数】从节点提取源码文本
        正确处理字节切片，防止中文乱码
        """
        if not node:
            return ""
        # 使用字节切片，然后解码回字符串
        return self.source_bytes[node.start_byte : node.end_byte].decode("utf8")

    def _find_and_process_classes(self, node, class_list, depth=0):
        """
        递归查找并处理类声明
        """
        if depth > 10: return # 防止过深

        # 遍历当前层级的子节点
        for child in node.children:
            if child.type == 'class_declaration':
                class_info = self._process_class_node(child)
                if class_info:
                    class_list.append(class_info)
            
            # 如果不是类定义，但可能包含类（比如模块定义等），可以在这里扩展
            # 对于内部类，我们会在 _process_class_node 内部递归处理，所以这里不需要深层递归

    def _process_class_node(self, class_node):
        """
        处理单个类节点
        """
        # 获取类名 (使用 field 'name')
        name_node = class_node.child_by_field_name('name')
        class_name = self._get_text(name_node)
        if not class_name: return None

        # 获取修饰符 (查找 modifiers 类型的子节点)
        modifiers = []
        modifiers_node = class_node.child_by_field_name('modifiers')
        if modifiers_node:
            for mod in modifiers_node.children:
                modifiers.append(self._get_text(mod))

        # 提取注解
        annotations = []
        # 遍历 class_node 的所有子节点
        for child in class_node.children:
            # 如果是 modifiers 节点，检查其内部子节点
            if child.type == 'modifiers':
                for mod_child in child.children:
                    # 检查是否是注解（包括 marker_annotation 和 annotation）
                    if mod_child.type in ['marker_annotation', 'annotation']:
                        annotations.append(self._get_text(mod_child))
        # 检查 modifiers 节点
        if modifiers_node:
            for child in modifiers_node.children:
                # 检查是否是注解（包括 marker_annotation 和 annotation）
                if child.type in ['marker_annotation', 'annotation']:
                    annotations.append(self._get_text(child))

        # 提取类继承关系和接口实现关系
        extends = []
        implements = []
        
        # 遍历 class_node 的所有子节点
        for child in class_node.children:
            # 查找 superclass 节点（父类）
            if child.type == 'superclass':
                # 提取父类名称，移除 'extends ' 前缀
                superclass_text = self._get_text(child)
                if superclass_text.startswith('extends '):
                    extends.append(superclass_text[8:].strip())
                else:
                    extends.append(superclass_text.strip())
            # 查找 super_interfaces 节点（实现的接口）
            elif child.type == 'super_interfaces':
                # 遍历 super_interfaces 节点的子节点，提取接口名称
                for interface_child in child.children:
                    if interface_child.type != 'comma' and interface_child.type != 'implements':  # 跳过逗号和 implements 关键字
                        interface_text = self._get_text(interface_child).strip()
                        if interface_text:
                            # 分割多个接口名（如果有逗号分隔）
                            for interface_name in interface_text.split(','):
                                trimmed_name = interface_name.strip()
                                if trimmed_name:
                                    implements.append(trimmed_name)

        class_info = {
            "name": class_name,
            "modifiers": modifiers,
            "annotations": annotations,
            "extends": extends,
            "implements": implements,
            "methods": [],
            "inner_classes": [],
            "original_code": self._get_text(class_node)
        }

        # 处理类体 (class_body)
        body_node = class_node.child_by_field_name('body')
        if body_node:
            for member in body_node.children:
                if member.type == 'method_declaration':
                    method_info = self._process_method_node(member, is_constructor=False)
                    class_info["methods"].append(method_info)
                
                elif member.type == 'constructor_declaration':
                    method_info = self._process_method_node(member, is_constructor=True)
                    class_info["methods"].append(method_info)
                
                elif member.type == 'class_declaration':
                    # 递归处理内部类
                    inner_class = self._process_class_node(member)
                    if inner_class:
                        class_info["inner_classes"].append(inner_class)

        return class_info

    def _process_method_node(self, method_node, is_constructor=False):
        """
        处理方法或构造函数节点
        """
        # 获取方法名
        name_node = method_node.child_by_field_name('name')
        method_name = self._get_text(name_node)

        # 返回类型 (构造函数没有返回类型)
        return_type = "void"
        if not is_constructor:
            type_node = method_node.child_by_field_name('type')
            if type_node:
                return_type = self._get_text(type_node)
            else:
                # 可能是 void_type，它不是一个 field，而是直接子节点
                for child in method_node.children:
                    if child.type == 'void_type':
                        return_type = "void"
                        break

        # 获取修饰符
        modifiers = []
        modifiers_node = method_node.child_by_field_name('modifiers')
        if modifiers_node:
            for mod in modifiers_node.children:
                modifiers.append(self._get_text(mod))

        # 参数提取
        parameters = []
        params_node = method_node.child_by_field_name('parameters')
        if params_node:
            for param in params_node.children:
                if param.type == 'formal_parameter':
                    # 提取 type 和 name
                    p_type = self._get_text(param.child_by_field_name('type'))
                    p_name = self._get_text(param.child_by_field_name('name'))
                    # 某些情况下（如 int... args）结构略有不同，这里做基础处理
                    if not p_type: 
                        # 尝试直接遍历寻找类型节点
                        for child in param.children:
                            if child.type.endswith('_type'): 
                                p_type = self._get_text(child)
                                break
                    
                    parameters.append({"type": p_type, "name": p_name})

        # 注释提取 (Looking backwards)
        comment = self._get_comments(method_node)

        # 收集方法调用
        called_functions = self._collect_invocations(method_node)

        # 提取注解
        annotations = []
        # 遍历 method_node 的所有子节点
        for child in method_node.children:
            # 如果是 modifiers 节点，检查其内部子节点
            if child.type == 'modifiers':
                for mod_child in child.children:
                    # 检查是否是注解（包括 marker_annotation 和 annotation）
                    if mod_child.type in ['marker_annotation', 'annotation']:
                        annotations.append(self._get_text(mod_child))
        # 检查 modifiers 节点
        if modifiers_node:
            for child in modifiers_node.children:
                # 检查是否是注解（包括 marker_annotation 和 annotation）
                if child.type in ['marker_annotation', 'annotation']:
                    annotations.append(self._get_text(child))

        return {
            "name": method_name,
            "is_constructor": is_constructor,
            "return_type": return_type,
            "modifiers": modifiers,
            "annotations": annotations,
            "parameters": parameters,
            "called_functions": called_functions,
            "comments": comment,
            "original_code": self._get_text(method_node)
        }

    def _get_comments(self, node):
        """
        获取节点上方的注释
        """
        comments = []
        curr = node.prev_sibling
        while curr:
            # Tree-sitter 中注释通常是 'line_comment' 或 'block_comment'
            # 注意：有时空格也会是兄弟节点，需要跳过
            if curr.type in ['line_comment', 'block_comment']:
                comments.insert(0, self._get_text(curr))
                curr = curr.prev_sibling
            elif curr.type == 'modifiers':
                # 如果遇到修饰符，说明注释在修饰符上面，继续往上找
                curr = curr.prev_sibling
            elif not curr.type.strip() or len(self._get_text(curr).strip()) == 0:
                # 跳过纯空白节点
                curr = curr.prev_sibling
            else:
                # 遇到其他代码结构，停止查找
                break
        return "\n".join(comments)

    def _collect_invocations(self, method_node):
        """
        收集方法体内的所有方法调用
        """
        invocations = set()
        
        body_node = method_node.child_by_field_name('body')
        if not body_node:
            # 可能是接口方法或抽象方法，没有 body
            return []

        # 递归查找 helper
        def traverse(node):
            if node.type == 'method_invocation':
                # 结构通常是: object.name(args) 或 name(args)
                name_node = node.child_by_field_name('name')
                if name_node:
                    func_name = self._get_text(name_node)
                    invocations.add(func_name)
            
            # 继续深度遍历所有子节点
            for child in node.children:
                traverse(child)

        traverse(body_node)
        return list(invocations)

    def analyze_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return self.analyze_code(f.read())

    def save_json(self, data, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Result saved to: {filepath}")

# --- 测试代码 ---
if __name__ == "__main__":
    analyzer = JavaCodeAnalyzer()
    
    # 分析javacodetest目录中的所有Java文件
    javacodetest_dir = "src/JavaCodeAnalyzer/javacodetest"
    
    if os.path.exists(javacodetest_dir):
        print(f"正在分析 {javacodetest_dir} 目录中的Java文件...\n")
        
        # 遍历目录中的所有.java文件
        for filename in os.listdir(javacodetest_dir):
            if filename.endswith(".java"):
                file_path = os.path.join(javacodetest_dir, filename)
                print(f"分析文件: {filename}")
                
                try:
                    # 分析文件
                    result = analyzer.analyze_file(file_path)
                    
                    # 保存结果
                    output_file = f"output/tree_sitter_{filename.replace('.java', '_analysis.json')}"
                    analyzer.save_json(result, output_file)
                    print()
                except Exception as e:
                    print(f"分析文件 {filename} 时出错: {e}")
    else:
        print(f"目录 {javacodetest_dir} 不存在")