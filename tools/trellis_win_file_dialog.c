#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <commdlg.h>

#include <stddef.h>

int trellis_win_open_image_file_dialog(char * out, size_t out_size) {
    if (out == NULL || out_size == 0) {
        return 0;
    }
    out[0] = '\0';

    wchar_t filename[32768];
    filename[0] = L'\0';

    OPENFILENAMEW ofn;
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFile = filename;
    ofn.nMaxFile = (DWORD) (sizeof(filename) / sizeof(filename[0]));
    ofn.lpstrTitle = L"Select input image";
    ofn.lpstrFilter = L"Images\0*.png;*.jpg;*.jpeg;*.webp;*.bmp\0All files\0*.*\0";
    ofn.nFilterIndex = 1;
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;

    if (!GetOpenFileNameW(&ofn)) {
        return 0;
    }

    int needed = WideCharToMultiByte(CP_UTF8, 0, filename, -1, NULL, 0, NULL, NULL);
    if (needed <= 0 || (size_t) needed > out_size) {
        return 0;
    }
    return WideCharToMultiByte(CP_UTF8, 0, filename, -1, out, (int) out_size, NULL, NULL) > 0;
}
