/**
 * 测试类继承关系和接口实现关系
 */
public class InheritanceTest {
    public static void main(String[] args) {
        // 测试代码
    }
}

/**
 * 基础父类
 */
class BaseClass {
    protected int value;
    
    public void setValue(int value) {
        this.value = value;
    }
    
    public int getValue() {
        return value;
    }
}

/**
 * 扩展基础父类
 */
class ExtendedClass extends BaseClass {
    private String name;
    
    public void setName(String name) {
        this.name = name;
    }
    
    public String getName() {
        return name;
    }
}

/**
 * 接口
 */
interface MyInterface {
    void doSomething();
    int getResult();
}

/**
 * 另一个接口
 */
interface AnotherInterface {
    void doAnotherThing();
}

/**
 * 实现单个接口
 */
class InterfaceImpl implements MyInterface {
    @Override
    public void doSomething() {
        System.out.println("Doing something");
    }
    
    @Override
    public int getResult() {
        return 42;
    }
}

/**
 * 实现多个接口
 */
class MultipleInterfaceImpl implements MyInterface, AnotherInterface {
    @Override
    public void doSomething() {
        System.out.println("Doing something");
    }
    
    @Override
    public int getResult() {
        return 42;
    }
    
    @Override
    public void doAnotherThing() {
        System.out.println("Doing another thing");
    }
}

/**
 * 继承并实现接口
 */
class ExtendedAndInterfaceImpl extends BaseClass implements MyInterface {
    @Override
    public void doSomething() {
        System.out.println("Doing something");
    }
    
    @Override
    public int getResult() {
        return value;
    }
}
