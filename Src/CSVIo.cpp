
#include "CSVIo.h"

#include <iostream>
#include <fstream>
#include <string>

using std::ios;
using std::wstring;
using std::wifstream;



/// @brief CSV文件解析器
class CCSVParser
{
public:
    /// @brief 构造函数
    explicit CCSVParser(IN const wchar_t* fileName)
    {
        m_fileName = fileName;
        m_bSkipHeader = false;
    }

    /// @brief 析构函数
    ~CCSVParser()
    {

    }

    /// @brief 设置是否跳过首行
    void SetSkipHeader(IN bool skip)
    {
        m_bSkipHeader = skip;
    }

    /// @brief 加载所有数据
    bool LoadAllData(OUT LDataMatrix& dataMatrix)
    {
        wstring str;
        wifstream fin(m_fileName, ios::in);

        // 文件不存在
        if (!fin) 
        {
            return false;
        }

        // 检查是否需要跳过首行
        if (m_bSkipHeader)
            getline(fin, str);

        while (getline(fin, str))
        {
        }

        return true;
    }

private:
    bool m_bSkipHeader; ///< 跳过首行
    wstring m_fileName; ///< 文件名

};

LCSVParser::LCSVParser(IN const wchar_t* fileName)
{
    m_pParser = nullptr;
    m_pParser = new CCSVParser(fileName);
}

LCSVParser::~LCSVParser()
{
    if (nullptr != m_pParser)
    {
        delete m_pParser;
        m_pParser = nullptr;
    }
}

void LCSVParser::SetSkipHeader(IN bool skip)
{
    m_pParser->SetSkipHeader(skip);
}

bool LCSVParser::LoadAllData(OUT LDataMatrix& dataMatrix)
{
    m_pParser->LoadAllData(dataMatrix);
}