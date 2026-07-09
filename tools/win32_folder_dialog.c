#ifdef _WIN32

#ifndef COBJMACROS
#define COBJMACROS
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <shobjidl.h>
#include <stddef.h>

int trellis_win32_select_folder(char * out, size_t out_size, const wchar_t * title) {
    if (out == NULL || out_size == 0) {
        return 0;
    }
    out[0] = '\0';

    HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    int needs_uninit = SUCCEEDED(hr);
    if (FAILED(hr) && hr != RPC_E_CHANGED_MODE) {
        return 0;
    }

    IFileOpenDialog * dialog = NULL;
    hr = CoCreateInstance(
        &CLSID_FileOpenDialog,
        NULL,
        CLSCTX_INPROC_SERVER,
        &IID_IFileOpenDialog,
        (void **) &dialog);
    if (FAILED(hr) || dialog == NULL) {
        if (needs_uninit) CoUninitialize();
        return 0;
    }

    DWORD options = 0;
    if (SUCCEEDED(IFileOpenDialog_GetOptions(dialog, &options))) {
        IFileOpenDialog_SetOptions(
            dialog,
            options | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST);
    }
    if (title != NULL && title[0] != L'\0') {
        IFileOpenDialog_SetTitle(dialog, title);
    }

    int ok = 0;
    hr = IFileOpenDialog_Show(dialog, NULL);
    if (SUCCEEDED(hr)) {
        IShellItem * item = NULL;
        hr = IFileOpenDialog_GetResult(dialog, &item);
        if (SUCCEEDED(hr) && item != NULL) {
            PWSTR path_w = NULL;
            hr = IShellItem_GetDisplayName(item, SIGDN_FILESYSPATH, &path_w);
            if (SUCCEEDED(hr) && path_w != NULL) {
                int n = WideCharToMultiByte(CP_UTF8, 0, path_w, -1, out, (int) out_size, NULL, NULL);
                if (n <= 0) {
                    n = WideCharToMultiByte(CP_ACP, 0, path_w, -1, out, (int) out_size, NULL, NULL);
                }
                ok = n > 0 && out[0] != '\0';
                CoTaskMemFree(path_w);
            }
            IShellItem_Release(item);
        }
    }

    IFileOpenDialog_Release(dialog);
    if (needs_uninit) CoUninitialize();
    return ok;
}

#endif
