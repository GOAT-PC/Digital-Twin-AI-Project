window.htconfig = {
    Style: {
        'texture.cache': true
    },
    Default: {
        // Resolve cross-domain issues
        crossOrigin: 'anonymous',
        convertURL: function (url) {
            if (/^.*\.mp4$/.test(url)) {
                return url;
            }
            if (/^data: /.test(url) || /^blob:/.test(url)) {
                return url;
            }

            var storagePrefix = /mini-browser/.test(window.location.href) ? '/a' : 'storage';
            if (storagePrefix && url && !/^data:image/.test(url) && !/^http/.test(url) && !/^https/.test(url)) {
                url = storagePrefix + '/' + url
            }
            return url;
        }
    },
    Project: {
        baseURL: 'http://127.0.0.1:9000', // 2024.12.6 新能源汽车baseUrl
        apiDelay: 1000,
        curveDelay: 6000,
    }
};
